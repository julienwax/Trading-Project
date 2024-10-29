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

seq_length = 25

### DATA ###

df = pd.read_csv("../data_lstm/DJIA_historical.csv")

### TRAINING FUNCTION ###

df.reset_index(inplace = True)
pd.set_option('display.max_columns', None)
df['Target'] = df['Close'] - df['Open']
df.dropna(inplace = True)
df.drop(['Close','High','Low','index','Target','Adj Close'], axis = 1, inplace = True)
sc = MinMaxScaler(feature_range=(0,1))
df_scaled = sc.fit_transform(df)  # data_set_scaled is a numpy array
print(df.head())
print(df.dtypes)
print(df.columns)
print(df_scaled.head())
print(df_scaled.dtypes)
print(df_scaled.columns)
length = df.shape[0]
data_size, vocab_size = len(df),1
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-3
N=len(df)//seq_length
X=np.zeros((N,seq_length,vocab_size))
Y=np.zeros((N,seq_length,vocab_size))
count=0
for p in range(0,len(df)-seq_length,seq_length):
    for l in range(seq_length):
        X[count][l][0]=df['Open'][p+l]
    for l in range(seq_length):
        Y[count][l][0]=df['Open' ][p+l+1]
    count += 1

print(X[0])
print(X[0].shape)
print(Y[0].shape)
print(Y[0])


def make_data(filename,seq_length=seq_length):
    df = pd.read_csv("../data_lstm/" + filename + ".csv")
    df.reset_index(inplace = True)
    df['Target'] = df['Close'] - df['Open']
    df['Target Class'] = [1 if df['Target'][i] > 0 else 0 for i in range(len(df))]
    df.dropna(inplace = True)
    df.drop(['Close','High','Low','Volume','index','Local time','Target'], axis = 1, inplace = True)
    X = []
    y = []
    for i in range(seq_length, df.shape[0]):
        X.append(df.iloc[i-seq_length:i,-1])
    for i in range(len(X)-1):
        y.append(X[i+1])
    X = X[:-1]
    X,y = np.array(X), np.array(y)
    return X,y

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0 and epoch != 0:
            print(" -- Epoch " + str(epoch) + " finished")
            pred=self.model.predict(X[0].reshape(1,seq_length,vocab_size),batch_size=1)
            #print("Prediction :", self.model.predict(X[batch].reshape(1,seq_length)))
            plt.plot(list(range(seq_length)),Y[0][:],label="data")
            plt.plot(list(range(seq_length)),pred[0][:],label="pred")
            plt.show()
        
def full_training(seq_length = seq_length):
    global X,Y,sc
    lstm_input = Input(batch_shape = (1,seq_length,1), name = "lstm_input")
    inputs = LSTM(150,name = "first_layer",stateful=True,return_sequences=True)(lstm_input)
    inputs = Dense(1,name = "dense_layer2",activation='relu')(inputs)
    output = Activation('linear', name = "output")(inputs)
    model = Model(inputs = lstm_input, outputs = output)
    adam = optimizers.Adam(10**(-3))
    model.compile(optimizer = adam, loss='mean_squared_error',metrics=['accuracy'])
    model.fit(x=X,y=Y, batch_size=1, epochs = 1000,validation_split=0.1,callbacks=[CustomCallback()])
    return model

def over_training(nb_epochs):
    global X,Y,sc,seq_length
    model = load_model("../trained_model/BoeingCo2-50.keras")
    model.fit(x=X,y=y, batch_size=1, epochs = nb_epochs,validation_split=0.1,callbacks=[CustomCallback()])
    model.save("../trained_model/BoeingCo2-50.keras")

def make_test(model,filename):
    global seq_length
    X,y = make_data(filename)
    splitlimit = int(0.9*len(X))
    X = X[splitlimit:]
    y = y[splitlimit:]
    y_pred = []
    model = load_model("../trained_model/BoeingCo2-50.keras")
    for i in range(len(X)):
        y_pred_i = model.predict(X[i].reshape(1,seq_length), verbose = 0)
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
    

model = full_training(seq_length)
model.save("../trained_model/BoeingCo2-50bis.keras")


#model = load_model("../trained_model/BoeingCo2-50.keras")
#make_test(model,"BA.USUSD_Candlestick_5_M_BID_27.04.2023-05.05.2024")

#over_training(5)