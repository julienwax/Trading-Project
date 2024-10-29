import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import pmdarima as pm
import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('MaunaLoaDailyTemps.csv',index_col='DATE'   ,parse_dates=True)
df=df.dropna()
#df = df.rename(columns = {'Unnamed: 0': 'date'})
#df1 = df['price']

def adf_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)

adf_test(df['AvgTemp'])
#stepwise_fit = auto_arima(df['AvgTemp'], trace=True,suppress_warnings=True)
train = df.iloc[:-30]; test = df.iloc[-30:]
print(train.shape)
print(test.shape)
model=ARIMA(train['AvgTemp'],order=(1,0,5))
model=model.fit()
print(model.summary())
start = len(train)
end = len(train) + len(test) - 1
pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA Predictions')
plt.plot(test.index, test['AvgTemp'], label='Actual')
plt.plot(test.index, pred, label='Predictions')
plt.legend()
mean_temp = test['AvgTemp'].mean()
print("Mean temperature in test data:", mean_temp)
rmse = sqrt(mean_squared_error(pred, test['AvgTemp']))
print("RMSE:", rmse)
plt.show()