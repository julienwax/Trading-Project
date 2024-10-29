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
import datetime as dt
"""
import yfinance as yf

def get_pe_ratio(ticker):
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    return info.get('trailingPE')

# Example usage:
print(get_pe_ratio('GOOGL'))  # Replace 'AAPL' with your ticker
"""

### PARAMETERS ###

backcandles = 800

### DATA ###

df = pd.read_csv("../data_lstm/BA.USUSD_Candlestick_5_M_BID_27.04.2023-05.05.2024.csv")

### TRAINING FUNCTION ###

### IL FAUT MODIFIER LE CODE POUR QUE Y SOIT COMPOSE DE SEQUENCE LE RESEAU DOIT ETRE ENTRAINE SUR DES SEQUENCES DE DONNEES ET NON PAS SUR DES DONNEES ISOLEES

df.reset_index(inplace = True)
pd.set_option('display.max_columns', None)
df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
df['Year'] = df['Local time'].apply(lambda x: x.strftime('%Y'))
df['Month'] = df['Local time'].apply(lambda x: x.strftime('%m'))
df['Day'] = df['Local time'].apply(lambda x: x.strftime('%d'))
df['Hour'] = df['Local time'].apply(lambda x: x.strftime('%H'))
df['Minute'] = df['Local time'].apply(lambda x: x.strftime('%M'))
df['Target'] = df['Close'] - df['Open']
df['Target Class'] = [1 if df['Target'][i] > 0 else 0 for i in range(len(df))]
df.dropna(inplace = True)
df.drop(['index','Local time','Target'], axis = 1, inplace = True)
sc = MinMaxScaler(feature_range=(0,1))
#print(df.head())
#print(df.dtypes)
#print(df.columns)
data_set_scaled = sc.fit_transform(df)  # data_set_scaled is a numpy array
X = []
for j in range(data_set_scaled.shape[1]-1):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i,j])
X = np.moveaxis(X,[0],[2])
X,yi = np.array(X), np.array(data_set_scaled[backcandles:,-1])
y = []
for i in range(backcandles, data_set_scaled.shape[0]):
    y.append(data_set_scaled[i-backcandles:i,-1])
y = np.array(y)
print(X.shape)
print(y.shape)

def make_data(filename,backcandles=backcandles):
    df = pd.read_csv("../data_lstm/" + filename + ".csv")
    df.reset_index(inplace = True)
    df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
    df['Year'] = df['Local time'].apply(lambda x: x.strftime('%Y'))
    df['Month'] = df['Local time'].apply(lambda x: x.strftime('%m'))
    df['Day'] = df['Local time'].apply(lambda x: x.strftime('%d'))
    df['Hour'] = df['Local time'].apply(lambda x: x.strftime('%H'))
    df['Minute'] = df['Local time'].apply(lambda x: x.strftime('%M'))
    df['Target'] = df['Close'] - df['Open']
    df['Target Class'] = [1 if df['Target'][i] > 0 else 0 for i in range(len(df))]
    df.dropna(inplace = True)
    df.drop(['index','Local time','Target'], axis = 1, inplace = True)
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(df)  # data_set_scaled is a numpy array
    X = []
    for j in range(data_set_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2])
    X = np.array(X)
    y = []
    for i in range(backcandles, data_set_scaled.shape[0]):
        y.append(data_set_scaled[i-backcandles:i,-1])
    y = np.array(data_set_scaled[backcandles:,-1])
    y = np.reshape(yi,(len(yi),1))
    return X,y,sc,data_set_scaled

class CustomCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 500 == 0 and batch != 0:
            print(" -- Batch " + str(batch) + " finished")
            
        
def full_training(backcandles = backcandles):
    global X,y,sc
    lstm_input = Input(batch_shape = (1,backcandles,X.shape[2]), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer",stateful=True,return_sequences=True)(lstm_input)
    inputs = Dense(1,name = "dense_layer2",activation='sigmoid')(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam()
    model.compile(optimizer = adam, loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x=X,y=y, batch_size=1, epochs = 1,validation_split=0.1,callbacks=[CustomCallback()])
    return model

def over_training(nb_epochs):
    global X,Y,sc,backcandles
    model = load_model("../trained_model/BoeingCo10-800.keras")
    model.fit(x=X,y=y, batch_size=1, epochs = nb_epochs,validation_split=0.1,callbacks=[CustomCallback()])
    model.save("../trained_model/BoeingCo10-800.keras")

def make_test(model,filename):
    global backcandles
    X,y,sc,data_set_scaled = make_data(filename)
    splitlimit = int(0.97*len(X))
    X = X[splitlimit:]
    y = y[splitlimit:]
    y = [x[0] for x in y]
    y = np.array(y)
    y_pred = []
    for i in range(len(X)):
        y_pred_i = model.predict(X[i].reshape(1,backcandles,data_set_scaled.shape[1]-1), verbose = 0)
        y_pred_i = y_pred_i.reshape(800,1)
        if i % 100 == 0:
            print(" -- Test " + str(100*i/len(X)) + " finished")
        y_pred.append(y_pred_i)
    y_pred = np.array(y_pred)
    y_pred = [y[0] for y in y_pred]
    y_pred = np.array(y_pred)
    data_pred_scaled = np.zeros(shape=(len(y_pred), data_set_scaled.shape[1]))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    print(y_pred_inversed)
    y_pred_inversed = [1 if i > 0.5 else 0 for i in y_pred_inversed]
    print(y_pred_inversed)
    print(np.sum(y_pred_inversed),len(y_pred_inversed)-np.sum(y_pred_inversed))
    print(np.sum(y),len(y)-np.sum(y))
    RMSE = np.sqrt(np.mean((y_pred - y) ** 2))
    print("RMSE =", RMSE)
    #print("y_pred_inversed =", y_pred_inversed)
    #print("y_inversed =", y_inversed)
    print(np.sum(y_pred_inversed==y),len(y_pred_inversed)-np.sum(y_pred_inversed==y))
    success_rate = np.mean([1 if y_pred_inversed[i] == y[i] else 0 for i in range(len(y))])
    print("Success rate =", success_rate)
    with open("../saved_logs/BoeingCo10-800.txt","a") as f:
        f.write(" Success rate = " + str(success_rate) + "\n")

#model = full_training(backcandles)
#model.save("../trained_model/BoeingCo10-800.keras")

model = load_model("../trained_model/BoeingCo10-800.keras")
make_test(model,"BA.USUSD_Candlestick_5_M_BID_27.04.2023-05.05.2024")

#over_training(4)