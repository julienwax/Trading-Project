### IMPORTATIONS ###
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from import_data import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Input,Activation,concatenate
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.models import load_model

### TICKER LIST ###
tickers_list = [
    'NVDA', # NVIDIA Corp
    'SOUN', # SoundHound AI Inc
    'UPST', # Upstart Holdings Inc
    'PRCT', # Procept BioRobotics Corp
    'AVAV', # AeroVironment Inc.
    'HLX',  # Helix Energy Solutions Group Inc
    'CRM',  # Salesforce
    'VERI', # Veritone Inc.
    'SPLK', # Splunk Inc.
    'MSFT', # Microsoft
    'TEAM', # Atlassian Corporation
    'PLTR', # Palantir Technologies Inc.
    'TTD',  # The Trade Desk, Inc.
    'BIDU', # Baidu, Inc.
    'ROK',  # Rockwell Automation, Inc.
    'SPLK', # Splunk Inc.
    'IRM',  # Iron Mountain Incorporated
    'TER',  # Teradyne, Inc.
    'EPAM', # EPAM Systems, Inc.
    'PATH', # UiPath Inc.
    'TWLO', # Twilio Inc.
    'ALAB', # Astera Labs, Inc.
    'TTEK', # Tetra Tech, Inc.
    'CGNX', # Cognex Corporation
    'SYM',  # Symbotic Inc.
    'AI',   # C3.ai, Inc.
    'AMBA', # Ambarella, Inc.
    'DOCU', # DocuSign Inc.
    'TSM',  # Taiwan Semiconductor Manufacturing Co. Ltd.
    'ABB',  # ABB Ltd.
    'NICE', # NICE Ltd.
    'DARK.L',           # Darktrace
    'PRESIGHT.AE',      # Presight AI
    'ODD',              # ODDITY Tech
    'NWTN',             # NWTN Inc.
    'RXRX',             # Recursion Pharmaceuticals
    'BAYANAT.AE',       # Bayanat AI
    'PRO',              # PROS
    'SOUN',             # SoundHound AI
    'NNOX',             # Nano-X Imaging
    'EXAI',             # Exscientia
    'CRNC',             # Cerence
    'BIGB',             # BigBear.ai
    # Additional tickers would continue here...
]


### CODES ###

backcandles = 50
splitlimit = 1

def make_data(ticker,backcandles,splitlimit):
    data = yf.download(tickers = ticker, start = '2013-03-11', end = "2023-03-22", interval = '1d')
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
    data_set = data.iloc[:,0:17]
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

def full_training(tickers_list = tickers_list, backcandles = backcandles, splitlimit = splitlimit):
    lstm_input = Input(shape = (backcandles,14), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer")(lstm_input)
    inputs = Dense(1,name = "dense_layer2")(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam()
    model.compile(optimizer = adam, loss='mse')
    a = None
    for ticker in tickers_list:
        print(ticker)
        try:
            X_train,X_test,y_train,y_test,sc,_ = make_data(ticker,backcandles,splitlimit)
            model.fit(x=X_train,y=y_train, batch_size=1,stateful = True, epochs = 5, validation_split=0.2)
            a = sc
        except Exception as e:
            print("Une erreur s'est produite avec ce ticker, erreur : " + str(e))
    return model,sc

def make_test(ticker,model):
    X_train,X_test,y_train,y_test,sc,data_set_scaled = make_data(ticker,backcandles,splitlimit)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
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
    plt.plot(y_pred_inversed, color = 'green', label = 'pred')
    plt.legend()
    RMSE = np.sqrt(np.mean((y_pred-y_test)**2))
    print("RMSE = " + str(RMSE))
    plt.show()

model,sc = full_training(tickers_list, backcandles, splitlimit)
model.save("trained_model/model14-50.keras")

#model = load_model("trained_model/model30.keras")
"""
X_train,X_test,y_train,y_test,sc,data_set_scaled = make_data("AAPL",backcandles,splitlimit)
print(X_train.shape)
print(y_train.shape)
print(X_train)
print(y_train)
print(data_set_scaled)


model = load_model("trained_model/model30.keras")

make_test("AMZN", model)
make_test("MSFT", model)
make_test("GOOGL", model)
make_test("AAPL", model)
make_test("MSFT", model)
make_test("INB", model)
make_test("QCOM", model)
"""
