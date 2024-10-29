import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

df = pd.read_csv('data_brute/META.PT1M.csv')
df1 = df['price']
df2 = pd.DataFrame({'price' : df1})
df2 = np.log(df2)

def prediction(i):
    global df2
    msk = (df2.index < i + 1)
    df_train = df2[msk].copy() # données d'entraînement
    df_test = df2[~msk].copy() # données de test
    model = ARIMA(df_train,order=(3,3,3))
    model_fit = model.fit()
    forecast_test = model_fit.forecast(1)
    #df['forecast_manual'] = [None]*(len(df_train)-1) + [df['Price'][len(df)-k-1]] + list(forecast_test)
    #df = np.exp(df)
    #df.plot()
    #plt.show()
    return np.exp(list(forecast_test)[0])


print(df[1000])
print(prediction(df,1000))