### Simple Bubble Strategy ###
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from backtesting import Strategy, Backtest
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
# ============================================================================
###  Bollinger Bands ###

#df = pd.read_csv("Data/NVDA_5MINS_13W_2024-08-05.csv")

def bollinger_bands(df,length=20,std=2):
    df.reset_index(drop=True,inplace=True)
    my_bbands = ta.bbands(df.Close, length=length, std=std)
    df = df.join(my_bbands, how='left')
    df.drop(columns=['BBB_20_2.0', 'BBP_20_2.0'], inplace=True)
    return df

#df = bollinger_bands(df)
pd.set_option('display.max_columns', None)
#print(df.head(20))

### Bubble Signal ###

def bubble_signal(df, current_candle):
    if max(df.Open[current_candle],df.Close[current_candle])<=df['BBL_20_2.0'][current_candle]:
        return 2
    if min(df.Close[current_candle],df.Open[current_candle])>=df['BBU_20_2.0'][current_candle]:
        return 1
    return 0

### VWA Signal ###

def vwap_signal(df, current_candle, backcandles=10):
    current_index = df.index.get_loc(current_candle)
    start = max(0, current_index - backcandles)
    end = current_index
    relevant_rows = df.iloc[start:end]
    if all(relevant_rows["Close"] < relevant_rows["VWAP_D"]):
        return 1
    elif all(relevant_rows["Close"] > relevant_rows["VWAP_D"]):
        return 2
    else:
        return 0

### Total Signal ###

def total_signal(df, current_candle,backcandles):
    if bubble_signal(df, current_candle)==2 and vwap_signal(df, current_candle,backcandles)==2:
        return 2
    if bubble_signal(df, current_candle)==1 and vwap_signal(df, current_candle,backcandles)==1:
        return 1
    return 0
        
def pointpos(x):
    if x['TotalSignal']==1:
        return x['High']*1.01
    elif x['TotalSignal']==2:
        return x['Low']*0.99
    else:
        return 0

# ============================================================================

def apply_save(filename,backtest=False):
    if backtest:
        df = pd.read_csv("Strategy/Data/"+filename+".csv")
    else:
        df = pd.read_csv("Data/"+filename+".csv")
    df=df[df['Volume']!=0]
    df.reset_index(drop=True,inplace=True)
    df = bollinger_bands(df)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.ta.vwap(append=True)
    df['ATR']=ta.atr(df.High, df.Low, df.Close, length=7)
    df['VWAP_Signal'] = df.apply(lambda row: vwap_signal(df, row.name,10), axis=1)
    df['BubbleSignal'] = df.apply(lambda row: bubble_signal(df, row.name), axis=1)
    df['TotalSignal'] = df.apply(lambda row: total_signal(df, row.name,10), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[20:]
    if backtest:
        df.to_csv("Strategy/Data/"+filename+".csv", index=False)
    else:
        df.to_csv("Data/"+filename+".csv", index=False)

apply_save("NVDA_5MINS_13W_2024-08-05")
#df = pd.read_csv("Data/NVDA_5MINS_13W_2024-08-05.csv")
#print(df.tail())
dfpl = None
# ============================================================================

def plot_signals(dfpl):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),

                go.Scatter(x=dfpl.index, y=dfpl['BBL_15_1.5'], 
                           line=dict(color='green', width=1), 
                           name="BBL"),
                go.Scatter(x=dfpl.index, y=dfpl['BBU_15_1.5'], 
                           line=dict(color='green', width=1), 
                           name="BBU"),
                go.Scatter(x=dfpl.index, y=dfpl['EMA_fast'], 
                           line=dict(color='black', width=1), 
                           name="EMA_fast"),
                go.Scatter(x=dfpl.index, y=dfpl['EMA_slow'], 
                           line=dict(color='blue', width=1), 
                           name="EMA_slow")])
    dfpl_filtered = dfpl[dfpl["pointpos"]!=0]
    fig.add_scatter(x=dfpl_filtered.index, y=dfpl_filtered["pointpos"], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="entry")
    fig.show()

#plot_signals(df)
#dfpl = df[100:]
#plot_signals(dfpl)

def SIGNAL():
    return dfpl['TotalSignal']

def run_strategy(df,strategy,filename ="",cash=100,margin=1/50,commission=.00,backtest=False):
    global dfpl
    dfpl = df
    bt = Backtest(df, strategy, cash=cash, margin=margin, commission=commission)
    stat = bt.run()
    print(stat)
    fig = bt.plot()
    current_path = os.getcwd()+"\\"
    if backtest:
        if not(os.path.exists("/Strategy/Backtest/"+strategy.__name__ + "_"+filename+".html")):
            shutil.copy(current_path+strategy.__name__ + ".html",current_path+"Strategy\\"+"Backtest\\" + strategy.__name__ + "_"+filename+".html")
    else:
        if not(os.path.exists("/Backtest/"+strategy.__name__ + "_"+filename+".html")):
            shutil.copy(current_path+strategy.__name__ + ".html",current_path+"Backtest\\" + strategy.__name__ + "_"+filename+".html")
    input("Press Enter to remove the HTML file...")
    if os.path.exists(strategy.__name__ + ".html"):
        os.remove(strategy.__name__ + ".html")

def optimize_strategy(df,strategy,filename="", cash=100, margin=1/50, commission=.00):
    bt = Backtest(df, strategy, cash=cash, margin=margin, commission=commission)
    stats,heatmap = bt.optimize(slcoef = [i/10 for i in range(10, 31)], TPSLRatio = [i/10 for i in range(10, 31)],
    maximize = 'Return [%]', max_tries = 500,random_state = 0, return_heatmap=True)
    print(stats)
    print(stats["_strategy"])
    heatmap_df = heatmap.unstack()
    plt.figure(figsize=(10, 8))
    plt.title(strategy.__name__ + " - " + filename)
    sns.heatmap(heatmap_df, cmap='viridis', annot=True, fmt=".0f")
    plt.savefig("C:\\Users\\Julien W\\Desktop\\artishow\\artishow\\Trading Bot\\Strategy\\Optimization_plot\\"+strategy.__name__ + "_"+filename+".png")
    plt.show()
    if os.path.exists(strategy.__name__ + ".html"):
        os.remove(strategy.__name__ + ".html")

class ScalpingBollinger_5min(Strategy):
    initsize = 0.9
    mysize = initsize
    commission = 0.00
    slcoef = 1.0
    TPSLRatio = 2.2
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        #df['RSI']=ta.rsi(df.Close, length=self.rsi_length)

    def next(self):
        super().next()
        slatr = self.slcoef*self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio

        # if len(self.trades)>0:                Fermetures des trades si le marché est en surachat ou survente
        #     if self.trades[-1].is_long and self.data.RSI[-1]>=90:
        #         self.trades[-1].close()
        #     elif self.trades[-1].is_short and self.data.RSI[-1]<=10:
        #         self.trades[-1].close()
        
        if self.signal1==2 and len(self.trades)==0:
            sl1 = self.data.Close[-1]*(1+self.commission) - slatr
            tp1 = self.data.Close[-1]*(1+self.commission) + slatr*TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)
        
        elif self.signal1==1 and len(self.trades)==0:         
            sl1 = self.data.Close[-1]*(1-self.commission) + slatr
            tp1 = self.data.Close[-1]*(1-self.commission) - slatr*TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

#run_strategy(dfpl,ScalpingBollinger_5min,filename ="EURUSD_5Min_2019_2022BB", cash=100, margin=1/50,commission=.00)
#optimize_strategy(dfpl,ScalpingBollinger, cash=100, margin=1/50)
