import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from math import ceil
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv('../data_lstm/DJIA_historical.csv') # (8918, 7)

def plot_close_data(df):
    close_data = df['Close'] 
    dates = df['Date'] 
    adj_dates = mdates.datestr2num(dates)
    plt.plot_date(adj_dates, close_data, '-', c ='darkblue')    
    plt.title('DJIA Close - Date, 1985-2020')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close value', fontsize = 16)
    plt.show()

#plot_close_data(df)

### PARAMETERS
splitlimit = 0.8
training_set = df.iloc[:ceil(splitlimit*len(df)), 5:6].values # (7135, 1)

# Scale training data
scaler = MinMaxScaler(feature_range = (0, 1))
scaler_test = MinMaxScaler(feature_range = (0, 1))
test_set = df.iloc[ceil(splitlimit*len(df)):, 5:6].values # (1783, 1)
training_set_scaled = scaler.fit_transform(training_set) 
test_set_scaled = scaler_test.fit_transform(test_set)

def plot_close_scaled_date(df, training_set_scaled):
    global splitlimit
    dates = df.iloc[:ceil(splitlimit*len(df)), 0:1].values 
    adj_dates = mdates.datestr2num(dates)
    plt.plot_date(adj_dates,training_set_scaled,'-',c='darkblue')    
    plt.title('DJIA Close Scaled - Date, 1985-2020')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close value', fontsize = 16)
    plt.show()

#plot_close_scaled_date(df, training_set_scaled)

# =============================================================================

vocab_size = 1

# hyperparameters
hidden_size = 200  # size of hidden layer of neurons
seq_length = 10 # number of steps to unroll the RNN for
learning_rate = 10e-4
N=len(training_set_scaled)//seq_length
N_test=len(test_set)//seq_length
X=np.zeros((N,seq_length,vocab_size))
Y=np.zeros((N,seq_length,vocab_size))
X_test=np.zeros((N_test,seq_length,vocab_size))
Y_test=np.zeros((N_test,seq_length,vocab_size))
count=0
epsilon=1e-5
for p in range(0,len(training_set_scaled)-seq_length,seq_length):
    for l in range(seq_length):
        X[count][l][0]=epsilon+(epsilon+training_set_scaled[p+l+1]-training_set_scaled[p+l])/(epsilon+training_set_scaled[p+l])
        Y[count][l][0]=epsilon+(epsilon+training_set_scaled[p+l+2]-training_set_scaled[p+l+1])/(epsilon+training_set_scaled[p+l+1])
    count += 1
count=0
for p in range(0,len(test_set_scaled)-seq_length,seq_length):
    for l in range(seq_length):
        X_test[count][l][0]=epsilon+(epsilon+test_set_scaled[p+l+1]-test_set_scaled[p+l])/(epsilon+test_set_scaled[p+l])
        Y_test[count][l][0]=epsilon+(epsilon+test_set_scaled[p+l+2]-test_set_scaled[p+l+1])/(epsilon+test_set_scaled[p+l+1])
    count += 1

print(X[0][0:seq_length][:])
print(X[0].shape)
print(Y[0].shape)
print(Y[0][0:seq_length][:])

#plt.plot(list(range(seq_length)),X[0][:], c = 'darkblue', label = "X[0]")
#plt.plot(list(range(seq_length)),Y[0][:], c = 'red', label = "Y[0]")
#plt.show()

def descale_data(X, original_data, epsilon=1e-5):
    global scaler
    scaled_data = []
    for i in range(len(X)):
        for j in range(seq_length):
            scaled_data.append(X[i][j][0])
    descaled_data = np.zeros_like(scaled_data)
    for i in range(1, len(scaled_data)):
        descaled_data[i] = (scaled_data[i] - epsilon) * (epsilon + original_data[i-1]) - epsilon + original_data[i-1]
    return scaler.inverse_transform(descaled_data.reshape(-1,1))

class MyCallback(Callback):
    def on_epoch_end(self,epoch, logs=None):
        if epoch%10==0:
            i = 0
            pred= model.predict(X[100].reshape(1,seq_length,vocab_size),batch_size=1)
            pred_test = model.predict(X_test[50].reshape(1,seq_length,vocab_size),batch_size=1)
            ax1=plt.subplot(1,2,1)
            ax2=plt.subplot(1,2,2)
            ax1.set_title("Training data")
            ax2.set_title("Test data")
            ax1.plot(list(range(seq_length)),Y[100][:],label="data",c = 'darkblue')
            ax1.plot(list(range(seq_length)),pred[0][:],label="pred",c = 'red')
            ax2.plot(list(range(seq_length)),Y_test[50][:],label="data_test",c = 'darkblue')
            ax2.plot(list(range(seq_length)),pred_test[0][:],label="pred",c = 'red')
            plt.show()



model = tf.keras.Sequential([
    tf.keras.Input(batch_shape=(1, seq_length, vocab_size)),
    tf.keras.layers.LSTM(hidden_size, activation='linear', return_sequences=True, stateful=True),
    tf.keras.layers.Dense(vocab_size, activation='linear')
])
model.compile(tf.keras.optimizers.Adam(),tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
model.fit(X,Y,batch_size=1,epochs=50, validation_split=0.1,callbacks=[MyCallback()])
model.save("../trained_model/DJ_model.keras")

"""
model = tf.keras.models.load_model("../trained_model/DJ_model1.keras")
pred= model.predict(X[0].reshape(1,seq_length,vocab_size),batch_size=1)
pred_test = model.predict(X_test[0].reshape(1,seq_length,vocab_size),batch_size=1)
print(pred[0][:])
print(pred_test[0][:])
ax1=plt.subplot(1,2,1)
ax2=plt.subplot(1,2,2)
ax1.set_title("Training data")
ax2.set_title("Test data")
ax1.plot(list(range(seq_length)),Y[100][:],label="data",c = 'darkblue')
ax1.plot(list(range(seq_length)),pred[0][:],label="pred",c = 'red')
ax2.plot(list(range(seq_length)),Y_test[20][:],label="data_test",c = 'darkblue')
ax2.plot(list(range(seq_length)),pred_test[0][:],label="pred",c = 'red')
plt.show()
"""