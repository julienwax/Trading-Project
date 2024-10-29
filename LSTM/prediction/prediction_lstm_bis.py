import sys
sys.path.insert(0, '..')  # to import LSTM_full.py
from training.LSTM_full import *
model = load_model("../trained_model/model14-50.keras")
sc = MinMaxScaler(feature_range=(0,1))

def download_df(ticker,start):
    df = yf.download(tickers = ticker, start = start)
    df['RSI'] = ta.rsi(df.Close,length = 15)
    df['EMAF'] = ta.ema(df.Close,length = 20)
    df['SMAF'] = ta.sma(df.Close,length = 20)
    df['SMAM'] = ta.sma(df.Close,length = 50)
    df['SMAS'] = ta.sma(df.Close,length = 150)
    df['ATR'] = ta.atr(df.High, df.Low,df.Close,length = 14)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['MACD_Histogram'] = ta.macd(df['Close'])['MACDh_12_26_9']
    df['MACD_Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
    df['Target'] = df['Adj Close']-df.Open
    df['Target'] = df['Target'].shift(-1)
    df.dropna(inplace = True)                           # seuls les 149 premières lignes sont supprimées
    return df

df = download_df("AMZN","2015-03-22")               # le df est de taille 14 est la dernière colonne est target
pd.set_option('display.max_columns', None)          # seulement pour créer X on ne regarde que les 13 premières colonnes
#print(df.columns)
#print(df.head())

def df_to_X(df,backcandles):
    df1 = df.copy()
    df1.drop(['Volume','Close'], axis = 1, inplace = True)
    #print(df1.shape)
    df_scaled = sc.fit_transform(df1)
    X = []
    for j in range(df_scaled.shape[1]-1):
        X.append([])
        for i in range(backcandles, df_scaled.shape[0]):
            X[j].append(df_scaled[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2]); X = np.array(X)
    return X

X = df_to_X(df,50)
print(X.shape)

def prediction_plot(df,backcandles=50,length=200):    # permet de prédire le prix le lendemain, X a le même format que les données d'entrainement
    X = df_to_X(df.iloc[-length:],backcandles)                            # scinde le df aux 200 derniers jours et prédit le prix du 201ème jour
    y_pred = model.predict(X)
    data_pred_scaled = np.zeros(shape=(len(y_pred), 15))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    plt.plot(y_pred_inversed, label='Predicted Price Difference', color = "darkblue")
    plt.plot(df['Target'].values[-length+backcandles:], label='Actual Price Difference', color = "magenta")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price Difference')
    plt.title("bakcandles = " + str(backcandles)+ " , length = " + str(length))
    plt.grid(True)
    rolling_mean = pd.Series(y_pred_inversed).rolling(window=20).mean()
    plt.plot(rolling_mean, label='Rolling Mean Predicted Price Difference', color='green')
    rolling_mean1 = pd.Series(df['Target'].values[-length+backcandles:]).rolling(window=20).mean()
    plt.plot(rolling_mean1, label='Rolling Mean Actual Price Difference', color='darkred')
    plt.axhline(y=np.mean(y_pred_inversed), color='r', linestyle='--', label='Mean Predicted Price Difference')
    plt.axhline(y=np.mean(df['Target'].values[-length+backcandles:]), color='b', linestyle='--', label='Mean Actual Price Difference')
    plt.legend()
    plt.show()

#prediction_plot(df,backcandles=50,length=200) 

def prediction(df,index,backcandles=50,length = 200):    # permet de prédire le prix le lendemain, l'index représente la date actuelle par un entier
    assert index >= length
    X = df_to_X(df.iloc[index-length:index+1],backcandles)                            
    y_pred = model.predict(X)
    data_pred_scaled = np.zeros(shape=(len(y_pred), 15))
    data_pred_scaled[:, -1] = y_pred.ravel()
    data_pred_inversed = sc.inverse_transform(data_pred_scaled)
    y_pred_inversed = data_pred_inversed[:, -1]
    return y_pred_inversed[-3:],df['Target'].values[index-3:index]

print(prediction(df,1000,50,200)) # on prédit le prix du 1001ème jour en se basant sur les 200 derniers jours
print(prediction(df,1500,50,200)) # on prédit le prix du 1501ème jour en se basant sur les 200 derniers jours
print(prediction(df,2000,50,200)) # on prédit le prix du 2001ème jour en se basant sur les 200 derniers jours
