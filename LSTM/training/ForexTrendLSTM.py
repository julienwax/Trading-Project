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

forex_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB",  # GAFAM
    "TSLA", "BRK-B", "V", "JNJ", "WMT",  # Tesla, Berkshire Hathaway, Visa, Johnson & Johnson, Walmart
    "JPM", "MA", "PG", "UNH", "DIS",  # JPMorgan Chase, MasterCard, Procter & Gamble, UnitedHealth, Disney
    "NVDA", "HD", "PYPL", "BAC", "VZ",  # NVIDIA, Home Depot, PayPal, Bank of America, Verizon
    "ADBE", "CMCSA", "NFLX", "INTC", "T",  # Adobe, Comcast, Netflix, Intel, AT&T
    "KO", "PFE", "CSCO", "PEP", "ABT",  # Coca-Cola, Pfizer, Cisco, PepsiCo, Abbott Laboratories
    "NKE", "CVX", "LLY", "MRK", "ABBV",  # Nike, Chevron, Eli Lilly, Merck, AbbVie
    "CRM", "ORCL", "ACN", "AVGO", "TXN",  # Salesforce, Oracle, Accenture, Broadcom, Texas Instruments
    "QCOM", "COST", "MDT", "BMY", "AMGN"  # Qualcomm, Costco, Medtronic, Bristol Myers Squibb, Amgen
]

### PARAMETERS ###

backcandles = 50

### TRAINING FUNCTION ###

def make_data(ticker,backcandles=backcandles):
    data = yf.download(tickers = ticker, start = '2015-03-11', end = "2024-04-27", interval = '1d')
    pd.set_option('display.max_columns', None)
    data['RSI'] = ta.rsi(data.Close,length = 15)
    data['SMAM'] = ta.sma(data.Close,length = 50)
    data['SMAS'] = ta.sma(data.Close,length = 150)
    data['VWAP'] = ta.vwap(data.High, data.Low, data.Close, data.Volume)
    data['Target'] = data['Close']-data['Open']
    data['Target'] = data['Target'].shift(-1)
    data['TargetClass'] = [1 if data.Target[i]> 0 else 0 for i in range(len(data))]
    data.dropna(inplace = True)
    data.reset_index(inplace = True)
    data.drop(['Date','Adj Close','Target'], axis = 1, inplace = True)
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data)  # data_set_scaled is a numpy array
    X = []
    for j in range(data_set_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2])
    X,yi = np.array(X), np.array(data_set_scaled[backcandles:,-1])
    y = np.reshape(yi,(len(yi),1))
    return X,y,sc,data_set_scaled

class CustomCallback(Callback):
    def on_batch_end(self,batch, logs=None):
        if batch % 500 == 0:
            print(" -- Batch " + str(batch) + " finished")

def full_training(tickers_list = forex_tickers, backcandles = backcandles):
    lstm_input = Input(batch_shape = (1,backcandles,9), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer",stateful=True)(lstm_input)
    inputs = Dense(1,name = "dense_layer2",activation='sigmoid')(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam()
    model.compile(optimizer = adam, loss='binary_crossentropy',metrics=['accuracy'])
    a = None
    for ticker in tickers_list:
        print(ticker)
        try:
            X,y,sc,_ = make_data(ticker,backcandles)
            model.fit(x=X,y=y, batch_size=1, epochs = 2,validation_split=0.2,callbacks=[CustomCallback()])
            a = sc
        except Exception as e:
            print("Une erreur s'est produite avec ce ticker, erreur : " + str(e))
    return model,a

### TESTING ###

def make_test(ticker,model):
    X_test,y_test,sc,data_set_scaled = make_data(ticker,backcandles)
    y_pred = []
    for i in range(len(X_test)):
        y_pred_i = model.predict(X_test[i].reshape(1,backcandles,data_set_scaled.shape[1]-1), verbose = 0)
        if i % 100 == 0:
            print(" -- Test " + str(100*i/len(X_test)) + " finished")
        if i < len(X_test)-1:
            taux = (y_test[i+1] - y_test[i])/y_test[i]
        y_pred.append(y_pred_i)
    y_pred = np.array(y_pred)
    data_pred_scaled = np.zeros(shape=(len(y_pred), data_set_scaled.shape[1]))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    y_pred_inversed = [1 if i > 0.5 else 0 for i in y_pred_inversed]
    data_test_scaled = np.zeros(shape=(len(y_test), data_set_scaled.shape[1]))
    data_test_scaled[:, -1] = y_test.ravel()
    data_test_inversed = sc.inverse_transform(data_test_scaled)
    y_test_inversed = data_test_inversed[:, -1]
    RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("RMSE =", RMSE)
    success_rate = np.mean([1 if y_pred_inversed[i] == y_test_inversed[i] else 0 for i in range(len(y_pred_inversed))])
    print("Success rate =", success_rate)
    with open("../saved_logs/forex9-50.txt","a") as f:
        f.write(str(ticker)+": Success rate = " + str(success_rate) + "\n")
    #plt.savefig("../saved_plot/"+str(ticker)+"-forex9-50"".png")

### MAIN ###

#model,sc = full_training(forex_tickers, backcandles)
#model.save("../trained_model/forex9-50.keras")


model = load_model("../trained_model/forex9-50.keras")

make_test("QCOM", model)
make_test("KO", model)
make_test("PEP", model)
make_test("NVDA", model)
make_test("TSLA", model)
make_test("FB", model)
make_test("ORCL", model)
make_test("NKE", model)
