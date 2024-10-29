### Scalping Strategy with RSI Indicator ###
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
### EMA Signal ###

def ema_signal(df,backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles-1, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.High[row]>=df.EMA200[row]:
                dnt=0
            if df.Low[row]<=df.EMA200[row]:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMAsignal'] = emasignal

### Total Signal ###

def total_signal(df):
    TotalSignal = [0] * len(df)
    for row in range(0, len(df)):
        TotalSignal[row] = 0
        if df.EMAsignal[row]==1 and df.RSI[row]>=90:
            TotalSignal[row]=1
        if df.EMAsignal[row]==2 and df.RSI[row]<=10:
            TotalSignal[row]=2
    df['TotalSignal']=TotalSignal
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

def pointpos(x):
    if x['TotalSignal']==1:
        return x['High']*1.01
    elif x['TotalSignal']==2:
        return x['Low']*0.99
    else:
        return 0

def apply_save(filename,backtest=False):
    if backtest:
        df = pd.read_csv("Strategy/Data/"+filename+".csv")
    else:
        df = pd.read_csv("Data/"+filename+".csv")
    df=df[df['Volume']!=0]
    df.reset_index(drop=True, inplace=True)
    df["EMA200"] = ta.ema(df.Close, length=200)
    df["RSI"] = ta.rsi(df.Close, length=3)
    df['ATR']= df.ta.atr(length=14)
    ema_signal(df,8)
    total_signal(df)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df = df[200:]
    if backtest:
        df.to_csv("Strategy/Data/"+filename+".csv", index=False)
    else:
        df.to_csv("Data/"+filename+".csv", index=False)

#apply_save("EURUSD_Candlestick_15_M_BID_31.01.2019-30.01.2022")
#df = pd.read_csv("Data/EURUSD_Candlestick_15_M_BID_31.01.2019-30.01.2022.csv")
#print(df.tail())
dfpl = None
# ============================================================================

def plot_signals(dfpl):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.EMA200, line=dict(color='red', width=1), name="EMA200")])

    dfpl_filtered = dfpl[dfpl["pointpos"]!=0]
    fig.add_scatter(x=dfpl_filtered.index, y=dfpl_filtered["pointpos"], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="entry")
    fig.show()

#plot_signals(df[0:2000])

#dfpl = df[200: 200+40000]

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
    

class ScalpingRSI_fixed(Strategy):
    initsize = 0.2
    mysize = initsize
    commission = 0.00
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        
        #if(self.signal1>0 and len(self.trades)==0 and len(self.closed_trades)>0 and self.closed_trades[-1].pl < 0):
            #self.mysize=min(1,self.mysize*2)
        #elif len(self.closed_trades)>0 and self.closed_trades[-1].pl > 0:
            #self.mysize=self.initsize

        if self.signal1==2 and len(self.trades)==0:   
            sl1 = self.data.Close[-1]*(1+self.commission) - 45e-4
            tp1 = self.data.Close[-1]*(1+self.commission) + 45e-4
            self.buy(sl=sl1, tp=tp1, size=self.mysize)
        
        elif self.signal1==1 and len(self.trades)==0:         
            sl1 = self.data.Close[-1]*(1-self.commission) + 45e-4
            tp1 = self.data.Close[-1]*(1-self.commission) - 45e-4 
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

#run_strategy(dfpl,ScalpingRSI_fixed)

# ============================================================================

class ScalpingRSI_15min(Strategy):  ## It is the Strategy used for the backtest file
    initsize = 0.9
    mysize = initsize
    commission = 0.00
    slcoef = 1.2
    TPSLRatio = 1.4
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()

        slatr = self.slcoef * self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio
        #if(self.signal1>0 and len(self.trades)==0 and len(self.closed_trades)>0 and self.closed_trades[-1].pl < 0):
            #self.mysize=min(1,self.mysize*2)
        #elif len(self.closed_trades)>0 and self.closed_trades[-1].pl > 0:
            # self.mysize=self.initsize
        if self.signal1 == 2 and len(self.trades) == 0:  
            sl1 = self.data.Close[-1]*(1 + self.commission) - slatr
            tp1 = self.data.Close[-1]*(1 + self.commission) + slatr * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)
        
        elif self.signal1 == 1 and len(self.trades) == 0:  
            sl1 = self.data.Close[-1]*(1 - self.commission) + slatr
            tp1 = self.data.Close[-1]*(1 - self.commission) - slatr * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

#run_strategy(dfpl,ScalpingRSI_15min,margin=1/50, commission=0.00)
#optimize_strategy(dfpl,ScalpingRSI_coef,margin=1/50, commission=0.0000)

# ============================================================================

class ScalpingRSI_trailing(Strategy):
    initsize = 0.2
    mysize = initsize
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        sltr = 1.5*self.data.ATR[-1]
        
        #if(self.signal1>0 and len(self.trades)==0 and len(self.closed_trades)>0 and self.closed_trades[-1].pl < 0):
        #    self.mysize=min(1,self.mysize*2)
        #elif len(self.closed_trades)>0 and self.closed_trades[-1].pl > 0:
        #    self.mysize=self.initsize

        for trade in self.trades: 
            if trade.is_long: 
                trade.sl = max(trade.sl or -np.inf, self.data.Close[-1] - sltr)
                if self.signal1==1:
                    trade.close()
            else:
                trade.sl = min(trade.sl or np.inf, self.data.Close[-1] + sltr) 
                if self.signal1==2:
                    trade.close()
                    
        if self.signal1==2 and len(self.trades)==0:
            sl1 = self.data.Close[-1] - sltr
            self.buy(sl=sl1, size=self.mysize)
        elif self.signal1==1 and len(self.trades)==0:
            sl1 = self.data.Close[-1] + sltr
            self.sell(sl=sl1, size=self.mysize)

#run_strategy(dfpl,ScalpingRSI_trailing,filename="EURUSD_Candlestick_15_M_BID_31.01.2019-30.01.2022",margin=1/50, commission=0.00)