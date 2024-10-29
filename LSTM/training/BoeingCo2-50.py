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
### FUTURES BESOINS POTENTIELS ###
# Le suréchantillonnage consiste à augmenter artificiellement le nombre d'exemples dans la classe minoritaire 
# en dupliquant des exemples existants ou en créant de nouveaux exemples synthétiques.

### PARAMETERS ###

backcandles = 10

### DATA ###

df = pd.read_csv("../data_lstm/BA.USUSD_Candlestick_5_M_BID_27.04.2023-05.05.2024.csv")

### TRAINING FUNCTION ###

df.reset_index(inplace = True)
pd.set_option('display.max_columns', None)
df['Target'] = df['Close'] - df['Open']
df['Target Class'] = [1 if df['Target'][i] > 0 else 0 for i in range(len(df))]
print(df['Target'].sum(), len(df['Target']) - df['Target'].sum())
df.dropna(inplace = True)
df.drop(['Close','High','Low','Volume','index','Local time','Target'], axis = 1, inplace = True)
sc = MinMaxScaler(feature_range=(0,1))
print(df.head())
print(df.dtypes)
print(df.columns)

X = []
y = []
for i in range(backcandles, df.shape[0]):
    X.append(df.iloc[i-backcandles:i,-1])
for i in range(len(X)-1):
    y.append(X[i+1])
X = X[:-1]
X,y = np.array(X), np.array(y)
print(X.shape)
print(y.shape)

def make_data(filename,backcandles=backcandles):
    df = pd.read_csv("../data_lstm/" + filename + ".csv")
    df.reset_index(inplace = True)
    df['Target'] = df['Close'] - df['Open']
    df['Target Class'] = [1 if df['Target'][i] > 0 else 0 for i in range(len(df))]
    df.dropna(inplace = True)
    df.drop(['Close','High','Low','Volume','index','Local time','Target'], axis = 1, inplace = True)
    X = []
    y = []
    for i in range(backcandles, df.shape[0]):
        X.append(df.iloc[i-backcandles:i,-1])
    for i in range(len(X)-1):
        y.append(X[i+1])
    X = X[:-1]
    X,y = np.array(X), np.array(y)
    return X,y

class CustomCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 2000 == 0 and batch != 0:
            print(" -- Batch " + str(batch) + " finished")
            print("Prediction :", self.model.predict(X[batch].reshape(1,backcandles)))
        
def full_training(backcandles = backcandles):
    global X,y,sc
    lstm_input = Input(batch_shape = (1,backcandles,1), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer",stateful=True,return_sequences=True)(lstm_input)
    inputs = Dense(1,name = "dense_layer2",activation='relu')(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam(10**(-3))
    model.compile(optimizer = adam, loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x=X,y=y, batch_size=1, epochs = 1,validation_split=0.1,callbacks=[CustomCallback()])
    return model

def over_training(nb_epochs):
    global X,Y,sc,backcandles
    model = load_model("../trained_model/BoeingCo2-50.keras")
    model.fit(x=X,y=y, batch_size=1, epochs = nb_epochs,validation_split=0.1,callbacks=[CustomCallback()])
    model.save("../trained_model/BoeingCo2-50.keras")

def make_test(model,filename):
    global backcandles
    X,y = make_data(filename)
    splitlimit = int(0.9*len(X))
    X = X[splitlimit:]
    y = y[splitlimit:]
    y_pred = []
    model = load_model("../trained_model/BoeingCo2-50.keras")
    for i in range(len(X)):
        y_pred_i = model.predict(X[i].reshape(1,backcandles), verbose = 0)
        if i % 100 == 0:
            print(" -- Test " + str(100*i/len(X)) + " finished")
        y_pred.append(y_pred_i)
    y_pred = np.array(y_pred)
    y_pred = [y_pred[i][-1,0] for i in range(y_pred.shape[0])]
    y_pred = [y[0] for y in y_pred]
    y_pred = [1 if y_pred[i] > 0 else 0 for i in range(len(y_pred))]
    y_pred = np.array(y_pred)
    y = [y[-1] for y in y]
    y = np.array(y)
    print(y_pred.shape)
    print(y_pred)
    print(y)
    print(y.shape)
    RMSE = np.sqrt(np.mean((y_pred - y) ** 2))
    print("RMSE =", RMSE)
    print(np.sum(y_pred==y),len(y_pred)-np.sum(y_pred==y))
    success_rate = np.mean([1 if y_pred[i] == y[i] else 0 for i in range(len(y))])
    print("Success rate =", success_rate)
    with open("../saved_logs/BoeingCo2-50.txt","a") as f:
        f.write(" Success rate = " + str(success_rate) + "\n")
    

model = full_training(backcandles)
model.save("../trained_model/BoeingCo2-50.keras")


#model = load_model("../trained_model/BoeingCo2-50.keras")
#make_test(model,"BA.USUSD_Candlestick_5_M_BID_27.04.2023-05.05.2024")

#over_training(5)