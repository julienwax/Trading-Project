### LSTM model for AAPL ###
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
import threading
# =============================================================================

df = pd.read_csv('../data_lstm/AAPL_15MINS_2Y_2024-06-25.csv') # (13016, 6)

### PARAMETERS ###
splitlimit = 0.8
hidden_size = 200 
seq_length = 15 
learning_rate = 10e-5
indicators = 2
### SPLIT OF DATA (80% TRAINING, 20% TESTING) ###

training_set = df.iloc[:int(splitlimit*len(df)), 4:5].values # (10412, 1)
test_set = df.iloc[int(splitlimit*len(df)):, 4:5].values # (2604, 1)

training_set_vol = df.iloc[:int(splitlimit*len(df)), 5:6].values # (10412, 1)
test_set_vol = df.iloc[int(splitlimit*len(df)):, 5:6].values # (2604, 1)

### SCALING DATA ###
scaler = StandardScaler()
scaler_vol = StandardScaler()

training_set_scaled = scaler.fit_transform(training_set) 
test_set_scaled = scaler.fit_transform(test_set)

training_set_vol_scaled = scaler_vol.fit_transform(training_set_vol)
test_set_vol_scaled = scaler_vol.fit_transform(test_set_vol)
### PLOTTING SCALED DATA ###

def plot_close_scaled(training_set_scaled):
    plt.plot(np.arange(training_set_scaled.shape[0]),[training_set_scaled[i][0] for i in range(training_set_scaled.shape[0])],'-',c='darkblue')    
    plt.title('NVDA Close Scaled | 2022-2024')
    plt.show()

#plot_close_scaled(training_set_scaled)

# =============================================================================

def make_vectors(data_scaled, seq_length=seq_length, indicators=1):
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

X_p,Y = make_vectors(training_set_scaled) # (694, 15, 1) both
X_vol,_ = make_vectors(training_set_vol_scaled) # (694, 15, 1) both
X_test_p,Y_test = make_vectors(test_set_scaled) # (173, 15, 1) both
X_test_vol,_ = make_vectors(test_set_vol_scaled) # (173, 15, 1) both

X = np.concatenate((X_p,X_vol),axis=2) # (694, 15, 2) | Y and Y_test has size (694, 15, 1) and (173, 15, 1) respectively
X_test= np.concatenate((X_test_p,X_test_vol),axis=2) # (173, 15, 2) 

### TRAINING ###

callback = False

class MyCallback(Callback):
    def on_epoch_end(self,epoch, logs=None):
        global callback
        if callback:
            """
            i,j = rd.randint(0,len(X)-1),rd.randint(0,len(X_test)-1)
            pred= model.predict(X[i].reshape(1,seq_length,indicators),batch_size=1)
            pred_test = model.predict(X_test[j].reshape(1,seq_length,indicators),batch_size=1)
            ax1=plt.subplot(2,2,1)
            ax2=plt.subplot(2,2,2)
            ax3=plt.subplot(2,2,3)
            ax4=plt.subplot(2,2,4)
            ax2=plt.subplot(1,2,2)
            ax1.set_title("Training data price variation")
            ax2.set_title("Test data price variation")
            ax1.plot(list(range(seq_length)),[Y[i][p][0] for p in range(seq_length)],label="data",c = 'darkblue')
            ax1.plot(list(range(seq_length)),[pred[0][p][0] for p in range(seq_length)],label="pred",c = 'red')
            ax2.plot(list(range(seq_length)),[Y_test[j][p][0] for p in range(seq_length)],label="data_test",c = 'darkblue')
            ax2.plot(list(range(seq_length)),[pred_test[0][p][0] for p in range(seq_length)],label="pred",c = 'red')
            plt.show()
            """
            for n in range(1, 9):
                if n % 2 != 0:  # Graphiques impairs pour l'entraînement
                    i = rd.randint(0, len(X)-1)
                    pred = model.predict(X[i].reshape(1, seq_length, indicators), batch_size=1)
                    ax = plt.subplot(4, 2, n)
                    ax.set_title(f"Training data price variation {n}")
                    ax.plot(list(range(seq_length)), [Y[i][p][0] for p in range(seq_length)], label="data", c='darkblue')
                    ax.plot(list(range(seq_length)), [pred[0][p][0] for p in range(seq_length)], label="pred", c='red')
                else:  # Graphiques pairs pour le test
                    j = rd.randint(0, len(X_test)-1)
                    pred_test = model.predict(X_test[j].reshape(1, seq_length, indicators), batch_size=1)
                    ax = plt.subplot(4, 2, n)
                    ax.set_title(f"Test data price variation {n}")
                    ax.plot(list(range(seq_length)), [Y_test[j][p][0] for p in range(seq_length)], label="data_test", c='darkblue')
                    ax.plot(list(range(seq_length)), [pred_test[0][p][0] for p in range(seq_length)], label="pred", c='red')
            plt.tight_layout()
            plt.show()
            callback = False
            if input("Do you want to save the model ? (y/n)") == "y":
                model.save(f"../trained_model/AAPLmodel_15min_{seq_length}_{indicators}.keras")
                print("Model saved")

def keyboard_listener():
    global callback; i = 0
    while True:
        input("Appuyez sur Entrée pour activer le callback à la prochaine époque...")
        i += 1
        if i % 2 == 0:
            callback = False
        else:
            callback = True

thread = threading.Thread(target=keyboard_listener)
thread.daemon = True
thread.start()

model = Sequential([
    Input(batch_shape=(1, seq_length, indicators)),
    LSTM(hidden_size, activation='tanh', return_sequences=True, stateful=True),
    Dropout(0.2),
    LSTM(hidden_size, activation='tanh', return_sequences=True, stateful=True),
    Dense(1, activation='linear')
])
model.compile(optimizer = optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError(),metrics=['accuracy'])
model.fit(X,Y,batch_size=1,epochs=1000, validation_split=0.1,callbacks=[MyCallback()])

#model.save("../trained_model/NVDAmodel.keras")
#model = load_model("../trained_model/NVDAmodel.keras")
