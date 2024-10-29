### IMPORTATIONS ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time,datetime,pytz
import sys
import re
import time
import json
import logging
import yfinance as yf
import matplotlib.pyplot as plt
from arima_model import prediction
from import_data import download1,existe_fichier

###

class Trading_bot():

    ### Initialisation

    def __init__(self,sim,ticker):
        
        self.sim=sim # booléen de simulation
        self.date=pd.to_datetime('2019-06-15') # temps initial
        self.end=pd.to_datetime('2024-02-01') # temps final
        self.open_market = True # Ouverture du marché

        self.ticker = ticker
        self.stocknames = [ticker]
        self.nasdaq_time_diff=6  # décalage horaire
        self.chartdict = {
                'P1D': {}, "INI":{}
            } # dictionnaires de clés ticker et de valeur une serie temporelle des prix (jours et minutes)
        self.trading_fee=2 # taxe de transaction
        self.logging = True
        self.historique_p = []
        self.historique_d = []
        self.capitaux = []

    ### Méthodes pratiques

    def now(self):                              # horloge   
        return self.date

    def sleep(self,jours):                       # décalage temporel en jours
        offset = pd.DateOffset(days = jours)    
        self.date=self.date+offset

    def get_latest_price(self,ticker): # Renvoie le dernier prix ou faux le marché est fermé
        df2=self.chartdict['P1D'][ticker]
        if (self.date in df2.index):
            return df2[self.date]
        else:
            if(self.open_market):
                print(self.date)
                print(self.now(),"Market Closed")
            return False

    def log(self,string,logf):
        if(self.logging):
            print(string,file=logf)

    def prepare_chart_dict(self):  # Initialise le dictionnaire de clés resolution et de valeur dictionnaire ticker -> séries temporelles
            for ticker in self.stocknames:
                df1=pd.read_csv('./data_brute/'+ticker+'.P1D.csv')
                df2=pd.Series(list(df1['price']),index=list(pd.to_datetime(df1.iloc[:,0])))  # Création d'une serie temporelle
                self.chartdict['P1D'][ticker]=df2
                self.chartdict['INI'][ticker]=df1

    ### Stratégie

    def buy_sell(self,total_avail,pos,logf):
        # total_avail: argent disponible dans le portefeuille
        # pos: nombre d"action dans le portefeuille
        # logf: fichier de log
        date=self.now()
        offset = pd.DateOffset(days = 30)
        start=date - offset
        end=date-pd.DateOffset(days=1) 
        df=self.chartdict['P1D'][self.ticker]
        df_range=df[start:end]
        price=pd.Series(list(df_range[:]),index=range(0,len(df_range),1))
        if(price.rolling(window=5).mean().iloc[-1]*1.01 > price.rolling(window=10).mean().iloc[-1]):
            p=self.get_latest_price(self.ticker)
            size=total_avail//p
            return 'BUY',self.ticker,p,size
        elif (pos!=0 and price.rolling(window=5).mean().iloc[-1] < price.rolling(window=10).mean().iloc[-1]*0.99):
            p=self.get_latest_price(self.ticker)
            size=pos
            return 'SELL',self.ticker,p,size
        else:
            return None,None,0,0 

    ### INTERFACE DE TRADING

    def pool(self,poolname,initial_sum,load_existing_pool=False):  # Crée le pool de données sous la forme d'un dictionnaire
        if(load_existing_pool):                                    # Prend le pool déja existant dans le fichier json si load_existing_pool est vérifié
            with open(poolname+'.json') as handle:      
                pooldict = json.loads(handle.read())
            initial_sum=pooldict['total']+pooldict['leftover']
        else:
            ### INITIALISATION DU POOL
            self.historique_p = []
            self.historique_d = []
            pooldict={}
            pooldict['active']=0                # Indique si un ordre est en cours et si oui de quel type
            pooldict['total']=0       # Valeur des actifs 
            pooldict['total_exp']=0      # Valeur des fond attendus aprés une transaction (prend en compte les frais de transactions)
            pooldict['orderid']=-1                     # Identifiant de l'ordre en cours
            pooldict['ticker']= self.ticker                        # Ticker de l'actif financier
            pooldict['pos']=0                           # Nombre d'action dans le portefeuille
            pooldict['leftover']=initial_sum                    # Argent non utilisé dans le pool
            with open(poolname+'.json', 'w') as f:      # Ecrit le pool dans le fichier json
                f.write(json.dumps(pooldict))

        pool_inactif = (pooldict['active']==0)
        ordre_en_attente = (pooldict['active']==1)
        logf=open(poolname+'.log', 'w')
        self.log("%s %s" % (self.now(),poolname),logf)  # Print dans le terminal la date et le nom du pool contenu dans le fichier log
        logf.close()
        while(ordre_en_attente or pool_inactif):     # Ordre inactif ou en cours
            logf=open(poolname+'.log', 'a+')    # Ouvre le fichier texte en mode append
            ticker=pooldict['ticker']
            print("\r",self.now(),end="  ")     # Affiche la date
            offset = pd.DateOffset(days=1)
            date = self.date - offset
            if self.date.month != date.month:
                p = self.get_latest_price(self.ticker)
                print("Total : %s" % (pooldict['pos']*p+pooldict['leftover']),end = " ") # Affiche le total des fonds (ie actifs + leftover)
            if(self.now()> self.end):
                print("Fin de la simulation ")
                return (100.0*(pooldict['total']+pooldict['leftover'])/initial_sum)  # Renvoie le ROI
            if(pooldict['active']==2):              # Ordre en attente d'éxécution
                latest_price=self.get_latest_price(ticker)
                if(latest_price):                       #Si le marché est ouvert et que le prix existe pour cette date
                    ordre_en_attente,tot_trade = True, pooldict['total_exp']-2*self.trading_fee  #Actualisation des variables (tot_trade : prix de la transaction net)
                    self.log("%s Ordre execute de %s pour %s" % (self.now(),ticker ,tot_trade),logf)
                    pooldict['active']=0 
                    
            else:
                pool_inactif,tot_trade=False,0.0  # Pas de transactions
                ordre_en_attente = True
            if pool_inactif:
                self.log(pooldict,logf)         # Affiche le pool dans le terminal
            elif(not ordre_en_attente and tot_trade!=0.0): # Un trade a été passé juste avant
                pooldict['active']=0                            # Redevient inactif
                pooldict['total']=tot_trade                     # Actualisation des actifs
                pooldict['total_exp']=tot_trade
                pooldict['orderid']=-1
                pooldict['ticker']=0
                pooldict['pos']=0
                pooldict['leftover']=0
                with open(poolname+'.json', 'w') as f:  #Ecriture de l'ordre dans le fichier json
                    f.write(json.dumps(pooldict))
                pool_inactif=True
            else:
                total_avail=pooldict['leftover']
                pos=pooldict['pos']
                ordre,ticker,price,size=self.buy_sell(total_avail,pos,logf)
                if(pooldict['active']==0 and ordre =="BUY"):  #Ordre inactif et achat
                    if(price*size < pooldict['leftover']):
                        buyorderid=self.now().isoformat()
                        print(" Ordre d'achat execute : " + str(size) + " à " + str(price) + " $" )
                        p = self.get_latest_price(self.ticker)
                        self.historique_p.append(p)
                        self.historique_d.append(self.now())
                        self.capitaux.append(pooldict['pos']*p+pooldict['leftover'])
                        print("Total : %s" % (pooldict['pos']*p+pooldict['leftover']))
                    else:
                        buyorderid=0
                        print("ici")
                        logf.close()
                        continue 
                    buy_order_pending,tot_buy_trade=False,price*size
                    self.log("%s : Ordre d'achat place  %s %s %f" % (self.now(),buyorderid,ticker,tot_buy_trade),logf)
                    pooldict['total']=tot_buy_trade
                    pooldict['ticker']=ticker
                    pooldict['orderid']=buyorderid
                    pooldict['pos']=size
                    pooldict['active']=1
                    pooldict['leftover']=total_avail-price*size
                    with open(poolname+'.json', 'w') as f:
                        f.write(json.dumps(pooldict))
                        if(self.open_market):
                            print(pooldict)
                elif(ordre=="SELL"):
                    if(size <= pooldict['pos']):  # On vérifie qu'on a ce que l'on vend
                        sellorderid=self.now().isoformat()
                        print(" Ordre de vente éxécuté  : " + str(size) + " à " + str(price) + " $" )
                        p = self.get_latest_price(self.ticker)
                        self.historique_p.append(p)
                        self.historique_d.append(self.now())
                        self.capitaux.append(pooldict['pos']*p+pooldict['leftover'])
                        print("Total : %s" % (pooldict['pos']*p+pooldict['leftover']))
                    if(sellorderid):
                        pooldict['active']=2
                        pooldict['total_exp']=price*size 
                        pooldict['orderid']=sellorderid
                        tot_trade = pooldict['total_exp']-2*self.trading_fee
                        pooldict['leftover']+=price*size
                        pooldict['total']=pooldict['pos']*price - size*price
                        pooldict['pos']-=size
                        with open(poolname+'.json', 'w') as f:
                            f.write(json.dumps(pooldict))
                            if(self.open_market):
                                print(pooldict)
                        pool_inactif=False
                        ordre_en_attente=True
                        self.log("%s : Ordre de vente place   %s %s %f " % (self.now(),sellorderid,ticker,price*size),logf)
                    else:
                        self.log("Impossible de placer l'ordre",logf)
                        print('deviennent false')
                        pool_inactif=False
                        ordre_en_attente=False   
            self.sleep(jours = 1)  # On avance d'une journée
        logf.close()
        return (100*(pooldict['total']+pooldict['leftover'])/initial_sum)  # On renvoie le ROI en fin de simulation

### EXECUTION ###

if __name__ == "__main__":
    ticker1 = input("Veuillez entrer un Ticker : ")
    somme_ini = int(input("Veuillez entrer un montant initial : "))
    if not existe_fichier(ticker1):
        download1(ticker1)
    bot=Trading_bot(sim=True,ticker = ticker1)  # Création d'un bot pour la simulation
    bot.prepare_chart_dict()                    # Initialisation du dictionnaire des données
    print(bot.pool("Ordre_courant",somme_ini,False))
    plt.subplot(2,1,2)
    df = pd.read_csv("data_brute/"+ticker1+".P1D.csv", index_col= "Unnamed: 0", parse_dates=True); df['price'].plot()
    plt.scatter(bot.historique_d,bot.historique_p,marker = '+',c = 'r')
    plt.subplot(2,1,1); df1 = pd.Series(bot.capitaux,bot.historique_d);df1.index = pd.to_datetime(df1.index).date
    df1.plot(marker = '+',c='darkblue'); plt.show()
