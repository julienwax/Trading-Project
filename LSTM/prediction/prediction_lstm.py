import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from backtesting import Strategy
from backtesting import Backtest
import sys
sys.path.insert(0, '..')  # to import LSTM_full.py
from training.LSTM_full import *
model = load_model("../trained_model/model30.keras")
sc = MinMaxScaler(feature_range=(0,1))

def download_df(ticker,start):
    df = yf.download(tickers = ticker, start = start)
    df['RSI'] = ta.rsi(df.Close,length = 15)
    df['EMAF'] = ta.ema(df.Close,length = 20)
    df['EMAM'] = ta.ema(df.Close,length = 100)
    df['EMAS'] = ta.ema(df.Close,length = 150)
    df['ATR'] = ta.atr(df.High, df.Low,df.Close,length = 14)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['MACD_Histogram'] = ta.macd(df['Close'])['MACDh_12_26_9']
    df['MACD_Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
    df['Target'] = df['Adj Close']-df.Open
    df['Target'] = df['Target'].shift(-1)
    df.dropna(inplace = True)                           # seuls les 149 premières lignes sont supprimées
    return df

df = download_df("GOOGL","2019-03-22")               # le df est de taille 14 est la dernière colonne est target
pd.set_option('display.max_columns', None)          # seulement pour créer X on ne regarde que les 13 premières colonnes
#print(df.columns)
#print(df.head())

def df_to_X(df,backcandles):
    df1 = df.copy()
    df1.drop(['Volume','Close'], axis = 1, inplace = True)
    df_scaled = sc.fit_transform(df1)
    X = []
    for j in range(df_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, df_scaled.shape[0]):
            X[j].append(df_scaled[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2]); X = np.array(X)
    return X

X = df_to_X(df,30)
print(X.shape)

def prediction_plot(df,backcandles=30,length=200):    # permet de prédire le prix le lendemain, X a le même format que les données d'entrainement
    X = df_to_X(df.iloc[-length:],backcandles)                            # scinde le df aux 200 derniers jours et prédit le prix du 201ème jour
    y_pred = model.predict(X)
    data_pred_scaled = np.zeros(shape=(len(y_pred), 14))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    plt.plot(y_pred_inversed, label='Predicted Price Difference', color = "darkblue")
    plt.plot(df['Target'].values[-length+backcandles:], label='Actual Price Difference', color = "magenta")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price Difference')
    plt.title("bakcandles = " + str(backcandles)+ " , length = " + str(length))
    plt.grid(True)
    rolling_mean = pd.Series(y_pred_inversed).rolling(window=20).mean()
    plt.plot(rolling_mean, label='Rolling Mean Predicted Price Difference', color='green')
    rolling_mean1 = pd.Series(df['Target'].values[-length+backcandles:]).rolling(window=20).mean()
    plt.plot(rolling_mean1, label='Rolling Mean Actual Price Difference', color='darkred')
    plt.axhline(y=np.mean(y_pred_inversed), color='r', linestyle='--', label='Mean Predicted Price Difference')
    plt.axhline(y=np.mean(df['Target'].values[-length+backcandles:]), color='b', linestyle='--', label='Mean Actual Price Difference')
    plt.legend()
    plt.show()

#prediction_plot(df,length=150) # le modèle prédit le prix du lendemain en fonction des 30 derniers jours

def prediction(df,index,backcandles=30,length = 200):    # permet de prédire le prix le lendemain, l'index représente la date actuelle par un entier
    assert index >= length
    X = df_to_X(df.iloc[index-length:index+1],backcandles)                            
    y_pred = model.predict(X)
    data_pred_scaled = np.zeros(shape=(len(y_pred), 14))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    return y_pred_inversed[-1]

#print(prediction(df,400)) 
#print(prediction(df,600)) 
#print(prediction(df,800)) 
"""
def test(df,index):
    U,V = prediction(df,index)
    win = 0
    for i in range(len(U)):
        if U[i]*V[i] > 0:
            win += 1
    return win/len(U)*100

s = 0
for i in range(400,800):
    rate = test(df,i)
    if i % 5 == 0:
        print("rate = " + str(rate) + "%")
    s += rate
print("average rate = " + str(s/400) + "%")
"""

dfpl = df[:].copy()
dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=2), name="EMA")])

fig.show()

def addsignal(df, backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.High[i]>=df.EMA[i]:
                dnt=0
            if df.Low[i]<=df.EMA[i]:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMASignal'] = emasignal

"""
def SIGNAL():
    return dfpl.ordersignal

class MyStrat(Strategy):
    initsize = 0.05
    ordertime=[]
    def init(self):
        super().init()
        self.signal = self.I(SIGNAL)

    def next(self):
        super().next()
        
        for j in range(0, len(self.orders)):
            if self.data.index[-1]-self.ordertime[0]>10:#days max to fulfill the order!!!
                self.orders[0].cancel()
                self.ordertime.pop(0)   
            
        if len(self.trades)>0:
            if self.data.index[-1]-self.trades[-1].entry_time>=10:
                self.trades[-1].close()
                #print(self.data.index[-1], self.trades[-1].entry_time)
            
            if self.trades[-1].is_long and self.data.RSI[-1]>=50:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1]<=50:
                self.trades[-1].close()
        
        if self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==2:
            #Cancel previous orders
            for j in range(0, len(self.orders)):
                self.orders[0].cancel()
                self.ordertime.pop(0)
            #Add new replacement order
            self.buy(sl=self.signal/2, limit=self.signal, size=self.initsize)
            self.ordertime.append(self.data.index[-1])
        
        elif self.signal!=0 and len(self.trades)==0 and self.data.EMASignal==1:
            #Cancel previous orders
            for j in range(0, len(self.orders)):
                self.orders[0].cancel()
                self.ordertime.pop(0)
            #Add new replacement order
            self.sell(sl=self.signal*2, limit=self.signal, size=self.initsize)
            self.ordertime.append(self.data.index[-1])

bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/10, commission=1.000)
stat = bt.run()
print(stat)
"""