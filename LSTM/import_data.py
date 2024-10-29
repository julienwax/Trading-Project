import yfinance as yf
import pandas as pd
import os

def f(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=6)
    df = pd.DataFrame()
    for i in range(4):
        df_bis = ticker.history(start = date2, interval="1m", end = date)
        df = pd.concat([df_bis,df])
        date = date2
        date2 = date - pd.DateOffset(days=7)
    df.to_csv("data_lstm/"+nom.upper()+".P1M.csv")
    print("Télécharger avec succès")

def g(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=729)
    df = ticker.history(start = date2, interval="1h", end = date)
    df.to_csv("data_lstm/"+nom.upper()+".P1H.csv")
    print("Télécharger avec succès")

def h(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=18000)
    df = ticker.history(period = "5y", interval="1d")
    df = df.drop(columns = ["Volume","Dividends","Stock Splits"], axis = 1)
    df = df.resample('D').ffill()
    df.index = pd.to_datetime(df.index).date
    df.to_csv("data_lstm/"+nom.upper()+".P1D.csv")
    print("Téléchargé avec succès")

def existe_fichier(ticker,dossier):
    chemin = dossier
    nom = ticker + ".P1D.csv"
    for nom_fichier in os.listdir(chemin):
        if nom_fichier == nom:
            return True
    return False

def download(ticker,freq):
    if not(existe_fichier(ticker,"data_lstm")):    
        ticker1 = yf.Ticker(ticker)
        data = ticker1.history(period="max")
        if not data.empty:
            if freq in {"m","h","d"}:
                if freq == "m":
                    f(ticker1,ticker)
                elif freq == "h":
                    g(ticker1,ticker)
                else:
                    h(ticker1,ticker)
            else:
                print("Fréquence %s invalide." % freq)
        else:
            print("Le ticker %s est invalide." % ticker)


