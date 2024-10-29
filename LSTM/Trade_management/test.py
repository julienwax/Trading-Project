import pandas as pd
import yfinance as yf

dfSPY = pd.read_csv("SPY.USUSD_Candlestick_1_D_BID_16.02.2017-21.05.2022.csv")

dfSPY=dfSPY[dfSPY.High!=dfSPY.Low]
dfSPY = dfSPY.reset_index()
dfSPY.head()
import pandas_ta as ta
dfSPY['EMA']=ta.sma(dfSPY.Close, length=200)#sma ema
dfSPY['RSI']=ta.rsi(dfSPY.Close, length=2)
#dfSPY.ta.indicators()
#help(ta.bbands)
my_bbands = ta.bbands(dfSPY.Close, length=20, std=2.5)
my_bbands[0:50]
dfSPY=dfSPY.join(my_bbands)
dfSPY.dropna(inplace=True)
dfSPY.reset_index(inplace=True)
#dfSPY.set_index('Gmt time', inplace=True)
dfSPY[420:425]

def addemasignal(df, backcandles):
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
            #print("!!!!! check trend loop !!!!")
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMASignal'] = emasignal

addemasignal(dfSPY, 3)

def addorderslimit(df, percent):
    ordersignal=[0]*len(df)
    for i in range(1, len(df)): #EMASignal of previous candle!!! modified!!!
        if df.EMASignal[i]==2 and df.Close[i]<=df['BBL_20_2.5'][i]:# and df.RSI[i]<=100: #Added RSI condition to avoid direct close condition
            ordersignal[i]=df.Close[i]-df.Close[i]*percent
        elif df.EMASignal[i]==1 and df.Close[i]>=df['BBU_20_2.5'][i]:# and df.RSI[i]>=0:
            ordersignal[i]=df.Close[i]+df.Close[i]*percent
    df['ordersignal']=ordersignal
    
addorderslimit(dfSPY, 0.01)

import numpy as np
def pointposbreak(x):
    if x['ordersignal']!=0:
        return x['ordersignal']
    else:
        return np.nan
dfSPY['pointposbreak'] = dfSPY.apply(lambda row: pointposbreak(row), axis=1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

dfpl = dfSPY[1000:1250].copy()
#dfpl=dfpl.drop(columns=['level_0'])#!!!!!!!!!!
#dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.EMA, line=dict(color='orange', width=2), name="EMA"),
                go.Scatter(x=dfpl.index, y=dfpl['BBL_20_2.5'], line=dict(color='blue', width=1), name="BBL_20_2.5"),
                go.Scatter(x=dfpl.index, y=dfpl['BBU_20_2.5'], line=dict(color='blue', width=1), name="BBU_20_2.5")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="Signal")
fig.show()

dfpl = dfSPY[:].copy()
def SIGNAL():
    return dfpl.ordersignal

from backtesting import Strategy
from backtesting import Backtest

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

bt = Backtest(dfpl, MyStrat, cash=10000, margin=1/100, commission=.000)
stat = bt.run()
print(stat)