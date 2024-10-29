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
    df.to_csv("data_brute/"+nom.upper()+".PT1M.csv")
    print("Télécharger avec succès")

def g(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=729)
    df = ticker.history(start = date2, interval="1h", end = date)
    df.to_csv("data_brute/"+nom.upper()+".PT1H.csv")
    print("Télécharger avec succès")

def h(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=18000)
    df = ticker.history(period = "5y", interval="1d")
    df = df.drop(columns = ["High","Low","Close","Volume","Dividends","Stock Splits"], axis = 1)
    df = df.resample('D').ffill()
    df = df.rename(columns={'Open': 'price'})
    df.index = pd.to_datetime(df.index).date
    df.to_csv("data_brute/"+nom.upper()+".P1D.csv")
    print("Téléchargé avec succès")

def h1(ticker,nom):
    date = pd.Timestamp.now().round(freq='min')
    date2 = date.floor('D')- pd.DateOffset(days=18000)
    df = ticker.history(period = "5y", interval="1d")
    df = df.drop(columns = ["High","Low","Close","Volume","Dividends","Stock Splits"], axis = 1)
    df = df.resample('D').ffill()
    df = df.rename(columns={'Open': 'price'})
    df.index = pd.to_datetime(df.index).date
    df.to_csv("nasdaq_data/"+nom.upper()+".P1D.csv")
    print("Téléchargé avec succès")

if __name__ == "__main__":
    reponse = input("Veuillez entrer un Ticker : ")
    ticker = yf.Ticker(reponse)
    data = ticker.history(period="max")
    if not data.empty:
        choice = input("Intervalle : m / h / d : ")
        if choice in {"m","h","d"}:
            if choice == "m":
                f(ticker,reponse)
            elif choice == "h":
                g(ticker,reponse)
            else:
                h(ticker,reponse)
        else:
            print("Choix non reconnu")
    else:
        print("Le Ticker n'est pas valide !!!")

def download():
    reponse = input("Veuillez entrer un Ticker : ")
    ticker = yf.Ticker(reponse)
    data = ticker.history(period="max")
    if not data.empty:
        choice = input("Intervalle : m / h / d : ")
        if choice in {"m","h","d"}:
            if choice == "m":
                f(ticker,reponse)
            elif choice == "h":
                g(ticker,reponse)
            else:
                h(ticker,reponse)
        else:
            print("Choix non reconnu")
    else:
        print("Le Ticker n'est pas valide !!!")

def download1(reponse):
    ticker = yf.Ticker(reponse)
    data = ticker.history(period="max")
    if not data.empty:
        choice = input("Intervalle : m / h / d : ")
        if choice in {"m","h","d"}:
            if choice == "m":
                f(ticker,reponse)
            elif choice == "h":
                g(ticker,reponse)
            else:
                h(ticker,reponse)
        else:
            print("Choix non reconnu")
    else:
        print("Le Ticker n'est pas valide !!!")

def download2(reponse):
    ticker = yf.Ticker(reponse)
    data = ticker.history(period="max")
    if not data.empty:
        h1(ticker,reponse)
    else:
        print("Le Ticker n'est pas valide !!!")

def existe_fichier(ticker):
    #dossier = input("Veuillez entrer un dossier de data: ")
    dossier = "data_brute"
    chemin = dossier
    nom = ticker + ".P1D.csv"
    for nom_fichier in os.listdir(chemin):
        if nom_fichier == nom:
            return True
    return False

def existe_fichier_nasdaq(ticker):
    #dossier = input("Veuillez entrer un dossier de data: ")
    dossier = "nasdaq_data"
    chemin = dossier
    nom = ticker + ".P1D.csv"
    for nom_fichier in os.listdir(chemin):
        if nom_fichier == nom:
            return True
    return False

