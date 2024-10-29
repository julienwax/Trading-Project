### LSTM model for NVDIA ###
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential, Input, losses, optimizers
import random as rd
# =============================================================================

df = pd.read_csv('../data_lstm/NVDA_15MINS_2Y_2024-06-23.csv') # (13016, 6)

### PARAMETERS ###
splitlimit = 0.8
hidden_size = 200 
seq_length = 15
learning_rate = 10e-4
indicators = 1
### SPLIT OF DATA (80% TRAINING, 20% TESTING) ###

training_set = df.iloc[:int(splitlimit*len(df)), 4:5].values # (10412, 1)
test_set = df.iloc[int(splitlimit*len(df)):, 4:5].values # (2604, 1)

### SCALING DATA ###

#scaler = MinMaxScaler(feature_range = (0, 1))
scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(training_set) 
test_set_scaled = scaler.fit_transform(test_set)

### PLOTTING SCALED DATA ###

def plot_close_scaled(training_set_scaled):
    plt.plot(np.arange(training_set_scaled.shape[0]),[training_set_scaled[i][0] for i in range(training_set_scaled.shape[0])],'-',c='darkblue')    
    plt.title('NVDA Close Scaled | 2022-2024')
    plt.show()

#plot_close_scaled(training_set_scaled)

# =============================================================================

def make_vectors(data_scaled, seq_length=seq_length, indicators=indicators):
    X = np.zeros((len(data_scaled)//seq_length, seq_length, indicators))
    Y = np.zeros((len(data_scaled)//seq_length, seq_length, 1))
    count = 0
    epsilon = 1e-5
    for p in range(0,len(data_scaled)-seq_length,seq_length):
        for l in range(seq_length):
            X[count][l][0] = (epsilon+data_scaled[p+l+1]-data_scaled[p+l])/(epsilon+data_scaled[p+l])
            Y[count][l][0] = (epsilon+data_scaled[p+l+2]-data_scaled[p+l+1])/(epsilon+data_scaled[p+l+1])
        count += 1
    return X, Y

X,Y = make_vectors(training_set_scaled) # (694, 15, 1) both
X_test,Y_test = make_vectors(test_set_scaled) # (173, 15, 1) both

class MyCallback(Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch % 3 ==0:
            i,j = rd.randint(0,len(X)-1),rd.randint(0,len(X_test)-1)
            pred= model.predict(X[i].reshape(1,seq_length,indicators),batch_size=1)
            pred_test = model.predict(X_test[j].reshape(1,seq_length,indicators),batch_size=1)
            ax1=plt.subplot(1,2,1)
            ax2=plt.subplot(1,2,2)
            ax1.set_title("Training data")
            ax2.set_title("Test data")
            ax1.plot(list(range(seq_length)),Y[i][:],label="data",c = 'darkblue')
            ax1.plot(list(range(seq_length)),pred[0][:],label="pred",c = 'red')
            ax2.plot(list(range(seq_length)),Y_test[j][:],label="data_test",c = 'darkblue')
            ax2.plot(list(range(seq_length)),pred_test[0][:],label="pred",c = 'red')
            plt.show()
            if input("Do you want to save the model ? (y/n)") == "y":
                model.save("../trained_model/NVDAmodel_15min.keras")
                print("Model saved")
            
"""
model = Sequential([
    Input(batch_shape=(1, seq_length, indicators)),
    LSTM(hidden_size, activation='tanh', return_sequences=True, stateful=True),
    Dense(indicators, activation='linear')
])
model.compile(optimizer = optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError(),metrics=['accuracy'])
model.fit(X,Y,batch_size=1,epochs=100, validation_split=0.1,callbacks=[MyCallback()])
"""
model = Sequential([
    Input(batch_shape=(1, seq_length, indicators)),
    LSTM(hidden_size, activation='tanh', return_sequences=True, stateful=True),
    Dropout(0.2),
    LSTM(hidden_size, activation='tanh', return_sequences=True, stateful=True),
    Dense(indicators, activation='linear')
])
model.compile(optimizer = optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError(),metrics=['accuracy'])
model.fit(X,Y,batch_size=1,epochs=100, validation_split=0.1,callbacks=[MyCallback()])


#=============================================================================
model.save("../trained_model/NVDAmodel_15min.keras")
#model = load_model("../trained_model/NVDAmodel.keras")

