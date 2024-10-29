### IMPORTATIONS ###
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from math import ceil
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Input,Activation,concatenate
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.models import load_model
import datetime as dt
### PARAMETERS ###
seq_length = 25
# =============================================================================
# Load data 
# =============================================================================

df = pd.read_csv("../data_lstm/DJIA_historical.csv")

#Extract close data and dates
close_data = df['Close'] # 1D (examples, )
dates = df['Date'] # 1D (examples, )
adj_dates = mdates.datestr2num(dates)

# Visualize whole dataset
plt.plot_date(adj_dates, close_data, '-')    
plt.title('DJIA Close vs. Date, 1985-2020')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close value', fontsize = 16)
plt.show()
    
# Define the training set
percent_training: float = 0.80
num_training_samples = ceil(percent_training*len(df)) # int
training_set = df.iloc[:num_training_samples, 5:6].values # (7135, 1)

# Scale training data
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set) #2D (num_training_samples, 1)
dates = df.iloc[:num_training_samples, 0:1].values # 1D (examples, )
adj_dates = mdates.datestr2num(dates)

# Visualize whole dataset

plt.plot_date(adj_dates,training_set_scaled)    
plt.title('DJIA Close vs. Date, 1985-2020')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close value', fontsize = 16)
plt.show()

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
data_size, vocab_size = len(training_set_scaled),2
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-3
N=len(training_set_scaled)//seq_length
X=np.zeros((N,seq_length,vocab_size))
Y=np.zeros((N,seq_length,vocab_size))
count=0
for p in range(0,len(training_set_scaled)-seq_length,seq_length):
    for l in range(seq_length):
        X[count][l][0]=0
        X[count][l][1]=0
        if(training_set_scaled[p+l+1] > training_set_scaled[p+l]):
            X[count][l][1]=1
        else:
            X[count][l][0]=1
    for l in range(seq_length):
        Y[count][l][0]=0
        Y[count][l][1]=0
        if(training_set_scaled[p+l+2] > training_set_scaled[p+l+1]):
            Y[count][l][1]=1
        else:
            Y[count][l][0]=1
    count += 1

print(X[0])
print(X[0].shape)
print(Y[0].shape)
print(Y[0])

class MyCallback(Callback):
  def on_batch_end(self,batch, logs=None):
    if batch%500==0:
        #pred=np.argmax(model.predict(X[0].reshape(1,seq_length,vocab_size),batch_size=1),axis=2)
        pred=model.predict(X[0].reshape(1,seq_length,vocab_size),batch_size=1)
        #print(pred)
        #print(Y[0])
        # Visualize whole dataset
        plt.plot(list(range(seq_length)),Y[0][:],label="data")
        plt.plot(list(range(seq_length)),pred[0][:],label="pred")
        #plt.plot(range(Y[0]),Y[0])    
        #plt.plot(range(pred),pred)    
        plt.legend(loc="upper left")
        plt.show()


lstm_input = Input(batch_shape=(1, seq_length, vocab_size), name="lstm_input")
inputs = LSTM(hidden_size, name="hidden_layer", stateful=True, return_sequences=True)(lstm_input)
inputs = Dense(vocab_size, name="dense_layer2", activation='softmax')(inputs)
output = Activation('linear', name="output")(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, batch_size=1, epochs=50, validation_split=0.1, callbacks=[MyCallback()])