
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os

from scipy.special import gamma

import matplotlib.pyplot as plt

# Funcion que retorna una lista con entradas que corresponden a los datos de un dia para una accion

def sep_date(bonddata):

    # Hacemos un array con las fechas
    days = bonddata["Dia"].drop_duplicates(keep="first").values

    # Creamos la lista donde guardaremos los dataframes diarios
    daily_dfs = []

    # Llenamos la lista
    for i in days:
        daily_dfs.append(bonddata.loc[bonddata["Dia"] == i])

    # Retornamos la lista
    return daily_dfs

# Funcion que resamplea los precios segun el intervalo de tiempo que nos interese

def set_freq(bonddata, freq):

    # Definimos date_time como indice
    #bonddata = bonddata.set_index("Datetime")

    # Resampleamos los precios segun la frecuencia
    price_resamp = bonddata.P_limpio.resample(freq).mean()
    price_resamp = price_resamp.between_time("8:00", "15:40")
    price_resamp = pd.DataFrame(price_resamp)

    return price_resamp

# Funcion que retorna un array de los retornos a una frecuencia dada

def get_returns(bonddailydata, freq, lag=1):

    # Resampleamos los precios segun la frecuencia escogida
    price_15min = set_freq(bonddailydata, freq)

    # Sacamos el logaritmo de los precios
    log_p = np.log(price_15min["P_limpio"].copy())

    # Retornamos la razon de los log-precios
    return log_p.diff(periods=lag).values

# Realized volatility

def RV(bonddailydata, freq, lag=1):

    # Obtenemos los retornos de los datos diarios
    ret = get_returns(bonddailydata, freq)

    # Retornamos la suma de los retornos cuadrados
    return np.nansum(np.square(ret))


# Bi-power variation

def BV(bonddailydata, freq):

    # Calculamos los retornos
    ret  = get_returns(bonddailydata, freq)

    # Definimos las constantes
    M    = len(ret) #len(stockdailydata)

    ###
    try:
        coef = (np.pi/2.)*float(M/(M-1.))
    except ZeroDivisionError:
        return np.nan

    #coef = (np.pi/2.)*float(M/(M-1.))
    ###

    # Inicializamos la lista que guarda los terminos de la sumatoria
    sums = []

    # Calculamos los terminos
    for i in range(2, M):
        sums.append(np.abs(ret[i] * ret[i-1]))

    # Retornamos la suma de los terminos que hay en la lista
    return coef*np.nansum(np.array(sums))


# Tri-power quarticity

def TP(bonddailydata, freq):

    # Calculamos los retornos absolutos
    bonddailyret = np.abs(get_returns(bonddailydata, freq))

    # Definimos las constantes
    M    = len(bonddailyret) #len(stockdailydata)
    mu43 = np.float_power(2., 2./3.)*gamma(7./6.)/gamma(0.5)
    ###
    try:
        coef = mu43*M*(M/(M-2))
    except ZeroDivisionError:
        return np.nan
    coef = mu43*M*(M/(M-2))
    ###
    p    = 4./3.

    # Inicializamos la lista donde guardaremos los terminos de la sumatoria
    sums = []

    # Calculamos los terminos de la sumatoria
    for i in range(3, M):
        temp1 = bonddailyret[i]*bonddailyret[i-1]*bonddailyret[i-2]
        temp2 = np.float_power(temp1, p)

        # Guardamos los terminos de la sumatoria en la lista sums
        sums.append(temp2)

    # Retornamos la suma-producto
    return coef*np.nansum(np.array(sums))


# Jump statistic

def JS(bonddata, freq):

    # Guardamos las fechas para indexar los datos al final
    days = bonddata["Dia"].drop_duplicates(keep="first").values

    # Inicializamos la lista donde se guardan los resultados
    js_result = []

    # Definimos las constantes
    vbb = (np.pi/2)**2 + np.pi - 3.
    vqq = 2.

    # Separamos nuestros datos por dias
    bonddailydata = sep_date(bonddata)

    # Hacemos el calculo de los valores diarios
    for df in bonddailydata:

        # Definimos las constantes DIARIAS
        #M = len(df)
        M = len(get_returns(df, freq))

        ###################print(M, df["Dia"][0])

        # Calculamos los parametros
        rv = RV(df, freq)
        bv = BV(df, freq)
        tp = TP(df, freq)

        ###
        #print("rv", rv, "bv", bv, "tp", tp)
        ###

        # Calculamos el numerador de la operacion
        num = np.log(rv) - np.log(bv)
        #num = rv - bv

        # Calculamos el denominador de la operacion
        ######
        try:
            temp = tp/(bv*bv)
            den  = np.sqrt( (vbb-vqq) * 1./M * max(1., temp) )
        except ZeroDivisionError:
            return np.nan
        #temp = tp/(bv*bv)

        #den  = np.sqrt( (vbb-vqq) * 1./M * max(1., temp) )
        ######

        # Guardamos el resultado en proporcion de superacion del valor critico 1.96
        try:
            _ = (num/den)/1.96
        except ZeroDivisionError:
            return np.nan
        js_result.append(_)

    # Verificamos si se supera el valor critico
    crit_val = (np.array(js_result) >= 1.).astype(int)

    # Retornamos el resultado
    return pd.DataFrame({"JS_statistic": js_result, "OverCritValue": crit_val}, index=days)
