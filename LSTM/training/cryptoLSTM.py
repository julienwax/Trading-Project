### IMPORTATIONS ###
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Input,Activation,concatenate
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.callbacks import Callback

### TICKER LIST ###

crypto_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
    "SOL-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
    "LTC-USD", "BCH-USD", "XLM-USD", "ALGO-USD", "VET-USD",
    "ETC-USD", "TRX-USD", "ATOM-USD", "XTZ-USD", "XMR-USD",
    "MIOTA-USD", "DASH-USD", "NEO-USD", "ZEC-USD", "WAVES-USD",
    "MKR-USD", "ONT-USD", "XEM-USD", "AAVE-USD", "COMP-USD",
    "SNX-USD", "DCR-USD", "ICX-USD", "KSM-USD", "SUSHI-USD",
    "YFI-USD", "QTUM-USD", "OMG-USD", "ZRX-USD", "LSK-USD",
    "BAL-USD", "REN-USD", "LRC-USD", "KNC-USD", "LUNA-USD",
    "UNI-USD", "FIL-USD", "THETA-USD", "BAT-USD"
]

### PARAMETERS ###

backcandles = 50
splitlimit = 1

### TRAINING FUNCTION ###

def make_data(ticker,backcandles,splitlimit):
    data = yf.download(tickers = ticker, start = '2020-03-11', end = "2024-04-27", interval = '1d')
    data['RSI'] = ta.rsi(data.Close,length = 15)
    data['EMAF'] = ta.ema(data.Close,length = 20)
    data['SMAF'] = ta.sma(data.Close,length = 20)
    data['SMAM'] = ta.sma(data.Close,length = 50)
    data['SMAS'] = ta.sma(data.Close,length = 150)
    data['ATR'] = ta.atr(data.High, data.Low,data.Close,length = 14)
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
    data['MACD_Histogram'] = ta.macd(data['Close'])['MACDh_12_26_9']
    data['MACD_Signal'] = ta.macd(data['Close'])['MACDs_12_26_9']
    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)
    data['TargetClass'] = [1 if data.Target[i]> 0 else 0 for i in range(len(data))]
    data["TargetNextClose"] = data['Adj Close'].shift(-1)
    data.dropna(inplace = True)
    data.reset_index(inplace = True)
    data.drop(['Volume','Close','Date'], axis = 1, inplace = True)
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)
    X = []
    for j in range(14):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2])
    X,yi = np.array(X), np.array(data_set_scaled[backcandles:,-3])
    y = np.reshape(yi,(len(yi),1))
    ind= int(len(X)*splitlimit)
    X_train, X_test = X[:ind],X[ind:]
    y_train,y_test = y[:ind],y[ind:]
    return X_train,X_test,y_train,y_test,sc,data_set_scaled

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f" -- Fin de l'époque {epoch+1}")



def full_training(tickers_list = crypto_tickers, backcandles = backcandles, splitlimit = splitlimit):
    lstm_input = Input(batch_shape = (1,backcandles,14), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer",stateful=True)(lstm_input)
    inputs = Dense(1,name = "dense_layer2")(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam()
    model.compile(optimizer = adam, loss='mse')
    a = None
    for ticker in crypto_tickers:
        print(ticker)
        try:
            X_train,X_test,y_train,y_test,sc,_ = make_data(ticker,backcandles,splitlimit)
            model.fit(x=X_train,y=y_train, batch_size=1, epochs = 5, validation_split=0.2,callbacks=[CustomCallback()])
            a = sc
        except Exception as e:
            print("Une erreur s'est produite avec ce ticker, erreur : " + str(e))
    return model,sc

### TEST FUNCTION ###

def make_test(ticker,model):
    _,X_test,_,y_test,sc,data_set_scaled = make_data(ticker,backcandles,0)
    y_pred = []
    for i in range(len(X_test)):
        y_pred_i = model.predict(X_test[i].reshape(1,backcandles,14), verbose = 0)
        y_pred.append(y_pred_i)
    y_pred = np.array(y_pred)
    data_pred_scaled = np.zeros(shape=(len(y_pred), data_set_scaled.shape[1]))
    data_pred_scaled[:, -3] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -3]
    data_test_scaled = np.zeros(shape=(len(y_test), data_set_scaled.shape[1]))
    data_test_scaled[:, -3] = y_test.ravel()
    data_test_inversed = sc.inverse_transform(data_test_scaled)
    y_test_inversed = data_test_inversed[:, -3]
    plt.figure(figsize =(16,8))
    plt.plot(y_test_inversed, color = 'black', label = 'Test')
    plt.plot(y_pred_inversed, color = 'green', label = 'Pred')
    plt.title(f"Ticker: {ticker}, Model: crypto14-50")
    plt.legend()
    RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("RMSE =", RMSE)
    plt.savefig("../saved_plot/"+str(ticker)+"-crypto14-50"".png")
    

########## MAIN ##########

#model,sc = full_training(crypto_tickers, backcandles, splitlimit)
#model.save("../trained_model/crypto14-50.keras")

model = load_model("../trained_model/crypto14-50.keras")

make_test("BTC-USD", model)
make_test("SOL-USD", model)
make_test("ZRX-USD", model)
make_test("LUNA-USD", model)
make_test("ADA-USD", model)
make_test("XLM-USD", model)
make_test("AAVE-USD", model)