### Swing LSTM Strategy ###
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
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# ============================================================================
### Loading the LSTM Model and scalers ###

backtest = True
model_name = "AAPLmodel_15min_15_2"

def set_model_name(name):
    global model_name
    model_name = name

def LoadModel():
    global model; global backtest; global model_name
    if backtest:
        model = load_model("LSTMmodel/trained_model/"+model_name+".keras")
    else:
        model = load_model("../LSTMmodel/trained_model/"+model_name+".keras")

LoadModel()
seq_length = 15
scaler = StandardScaler()
scaler_vol = StandardScaler()

### EMA Signal ###

def ema_signal(df,backcandles):
    emasignal = [0]*len(df)
    for row in range(backcandles-1, len(df)):
        upt = 1
        dnt = 1
        for i in range(row-backcandles, row+1):
            if df.High[row]>=df.EMA30[row]:
                dnt=0
            if df.Low[row]<=df.EMA30[row]:
                upt=0
        if upt==1 and dnt==1:
            emasignal[row]=3
        elif upt==1:
            emasignal[row]=2
        elif dnt==1:
            emasignal[row]=1
    df['EMAsignal'] = emasignal

### LSTM Signal ###

def LSTM_signal(scaled_close_var,scaled_vol, current_candle, model):
    if current_candle < seq_length:
        return 0
    X_p = scaled_close_var[current_candle-seq_length:current_candle]
    X_vol = scaled_vol[current_candle-seq_length:current_candle]
    X_p.reshape(-1,1); X_vol.reshape(-1,1)
    X_p = scaler.fit_transform(X_p); X_vol = scaler_vol.fit_transform(X_vol)
    X_p = X_p.reshape(1, X_p.shape[0], 1); X_vol = X_vol.reshape(1, X_vol.shape[0], 1)
    X = np.concatenate((X_p, X_vol), axis=2)
    pred = model.predict(X)
    if pred[0][-1][0] > 0.2:
        return 2
    elif pred[0][-1][0] < -0.2:
        return 1
    else:
        return 0

### Total Signal ###

def total_signal(df):
    TotalSignal = [0] * len(df)
    for row in range(0, len(df)):
        TotalSignal[row] = 0
        if df.LSTMsignal[row]==1 and df.EMAsignal[row]==1:
            TotalSignal[row]=1
        if df.LSTMsignal[row]==2 and df.EMAsignal[row]==2:
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

### Apply and Save ###

def apply_save(filename,backtest=False):
    if backtest:
        df = pd.read_csv("Strategy/Data/"+filename+".csv")
    else:
        df = pd.read_csv("Data/"+filename+".csv")
    df=df[df['Volume']!=0]
    scaled_close = df['Close'].values
    scaled_close = scaler.fit_transform(scaled_close.reshape(-1,1))
    scaled_close_var = [(1e-5+scaled_close[i+1][0]-scaled_close[i][0])/(scaled_close[i][0]+1e-5) for i in range(len(scaled_close)-1)]
    scaled_close_var = np.array(scaled_close_var)
    scaled_close_var = scaled_close_var.reshape(-1,1)
    scaled_vol = df['Volume'].values
    scaled_vol = scaler_vol.fit_transform(scaled_vol.reshape(-1,1))
    df.drop(0, axis=0, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df['RSI']=ta.rsi(df.Close, length=10)
    df['ATR']=ta.atr(df.High, df.Low, df.Close, length=7)
    df['EMA200'] = ta.ema(df.Close, length=200)
    df['EMA30'] = ta.ema(df.Close, length=30)
    ema_signal(df,8)
    df['LSTMsignal'] = df.apply(lambda row: LSTM_signal(scaled_close_var,scaled_vol, row.name, model) , axis=1)
    total_signal(df)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[200:]
    if backtest:
        df.to_csv("Strategy/Data/"+filename+".csv", index=False)
    else:
        df.to_csv("Data/"+filename+".csv", index=False)

#apply_save("AAPL_15MINS_2Y_2018-01-01")
#df = pd.read_csv("Data/AAPL_15MINS_2Y_2018-01-01.csv")
#dfpl = df[11000:]
dfpl = None
# ============================================================================
### Plotting the Signals ###

def plot_signals(dfpl):
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),])

    dfpl_filtered = dfpl[dfpl["pointpos"]!=0]
    fig.add_scatter(x=dfpl_filtered.index, y=dfpl_filtered["pointpos"], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="entry")
    fig.show()

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
    stats,heatmap = bt.optimize(slcoef = [i/5 for i in range(5, 26)], TPSLRatio = [i/5 for i in range(5, 26)],
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

class SwingLSTM_15min(Strategy):
    initsize = 0.9
    mysize = initsize
    commission = 0.00
    slcoef = 1.6
    TPSLRatio = 1.6
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        #df['RSI']=ta.rsi(df.Close, length=self.rsi_length)

    def next(self):
        super().next()
        slatr = self.slcoef*self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio

        #if len(self.trades)>0:               
        #    if self.trades[-1].is_long and self.data.RSI[-1]>=80:
        #        self.trades[-1].close()
        #    elif self.trades[-1].is_short and self.data.RSI[-1]<=20:
        #        self.trades[-1].close()
        
        if self.signal1==2 and len(self.trades)==0:
            sl1 = self.data.Close[-1]*(1+self.commission) - slatr
            tp1 = self.data.Close[-1]*(1+self.commission) + slatr*TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)
        
        elif self.signal1==1 and len(self.trades)==0:         
            sl1 = self.data.Close[-1]*(1-self.commission) + slatr
            tp1 = self.data.Close[-1]*(1-self.commission) - slatr*TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

#run_strategy(dfpl,SwingLSTM_15min,filename ="AAPL_15MINS_2Y_2024-06-25", cash=100, margin=1/10,commission=.00)
#optimize_strategy(dfpl,SwingLSTM_15min,filename ="AAPL_15MINS_2Y_2024-06-25", cash=100, margin=1/10)
