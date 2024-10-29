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
from import_data import download,existe_fichier

###

"""
bitcoin = pd.read_csv('BTC-EUR.csv',index_col='Date',parse_dates= True)
ethereum = pd.read_csv('ETH-EUR.csv',index_col='Date',parse_dates= True)
print(bitcoin.head())
print(bitcoin.index) 
btc_eth = pd.merge(bitcoin,ethereum, on = 'Date', how ='inner', suffixes = ('_btc','_eth'))
btc_eth[['Close_btc','Close_eth']].plot(subplots = True)
#bitcoin.loc['2021-07':'2023-02','Close'].plot()
#bitcoin.loc['2020','Close'].rolling(window=7).mean() moyenne glissante

plt.show()
"""

###

class Trading_bot():

    ### Initialisation

    def __init__(self,sim,ticker):
        
        self.sim=sim # booléen de simulation
        self.time=pd.to_datetime('2022-09-15 15:30:00') # temps initial
        self.end=pd.to_datetime('2024-02-01 15:30:00') # temps final
        self.open_market = True # Ouverture du marché

        self.ticker = ticker
        self.stocknames = [ticker]
        self.nasdaq_time_diff=6  # décalage horaire
        self.chartdict = {
                'P1D': {}, 
                'PT1M': {} 
            } # dictionnaires de clés ticker et de valeur une serie temporelle des prix (jours et minutes)
        self.trading_fee=2 # taxe de transaction
        self.logging = True

    ### Méthodes pratiques

    def now(self):                              # horloge   
        return self.time

    def sleep(self,secs):                       # décalage temporel en secondes
        offset = pd.DateOffset(minutes=secs//60)    
        self.time=self.time+offset

    def get_latest_price(self,ticker): # Renvoie le dernier prix ou faux le marché est fermé
        df2=self.chartdict['PT1M'][ticker]
        if (self.time in df2.index):
            return df2[self.time]
        else:
            if(self.open_market):
                print(self.time)
                print(self.now(),"Market Closed")
            return False

    def is_summertime(self,date):  #détermine si la date correspondant au fuseau estival
        timezone = pytz.timezone('America/New_York') #fuseau horaire de nasdaq et ny
        aware_dt = timezone.localize(date)  #détermine le fuseau horaire de la date
        return aware_dt.dst() != datetime.timedelta(0,0)

    def is_market_open(self):
        df2=self.chartdict['PT1M'][self.stocknames[0]]
        if (self.time in df2.index):
            if(self.is_summertime(self.time)): #heure d'été
                start=datetime.datetime(self.time.year,self.time.month,self.time.day,15,30)  #la bourse nasdaq ouvre a 15h30 à Paris car à 9h30 a NY
                if(self.time > start):
                    return True
                else:
                    return False
            else:       # heure d'hiver
                start=datetime.datetime(self.time.year,self.time.month,self.time.day,14,30) #la bourse nasdaq ouvre a 14h30 en hiver
                if(self.time > start):
                    return True
                else:
                    return False
        else:
            return False

    def is_market_open_at(self,time): #détermine si le marché était ouvert au temps time
        df2=self.chartdict['PT1M'][self.stocknames[0]]
        if (time in df2.index):
            return True
        else:
            return False

    def is_closing_hour(self):  # détermine si le marché ferme dans une minute
        if(self.sim):
            offset = pd.DateOffset(minutes=1)
            time=self.time+offset
            df2=self.chartdict['PT1M'][self.stocknames[0]]
            if (time in df2.index):
                return False
            else:
                return True

    def log(self,string,logf):    # Ecrit dans le terminal le contenu du fichier logf
        if(self.logging):
            print(string,file=logf)

    def curate(self,df,ticker): #filtre les données avec le dataframe df
        newdates = []       
        price = []
        count = 0 
        if df.shape[1] > 2:                              # Si il y a plus de 2 colonnes
            df = df.drop(['Open','High','Low','Adj Close','Volume'],axis = 1) # garde que la colonne prix et date
        errdate = None
        for date in list(pd.to_datetime(df.iloc[:,0])):  
            if(int(str(date.hour)) == 14 and (int(str(date.minute)) == 30) or (int(str(date.minute)) == 31)):
                errdate=[date.year,date.month,date.day]
            if(int(str(date.hour)) == 15 and (int(str(date.minute)) == 30) or (int(str(date.minute)) == 29) or (int(str(date.minute)) == 28)):
                pass
            elif int(str(date.hour)) == 22 and (int(str(date.minute)) == 00):
                pass
            if(errdate):
                if(errdate[0]==date.year and errdate[1]==date.month and errdate[2]==date.day):
                    newdate=date+pd.DateOffset(minutes=60)
                    newdates.append(newdate)
                    price.append(df.iloc[count,1])  
                else:
                    errdate=None 
                    newdates.append(date)
                    price.append(df.iloc[count,1])
            count+=1
        i = 0
        while (i < (len(newdates)-1)):

            if( newdates[i].year == newdates[i+1].year and
                newdates[i].month == newdates[i+1].month and
                newdates[i].day == newdates[i+1].day):
                diff=(newdates[i+1] -newdates[i]).total_seconds()//60
                diff=int(diff)                                  # différence en minutes entre deux dates consécutives
                if (diff!=1):
                    missing_prices=[]
                    missing_dates=[]
                    for j  in  range(diff-1):
                        newdates.insert(i+j+1,newdates[i]+pd.DateOffset(minutes=j+1))  #ajoute les dates manquantes
                        price.insert(i+j+1,price[i]+(j+1)*(price[i+1]-price[i])/diff) # ajout des prix à croissance linéaire
                    i+=diff
                else:
                    i+=1
            else:
                i+=1
            df2=pd.Series(price,index=newdates,name='price')  # Création du nouveau data frame
            df2.to_csv('./data_curated/'+ticker+'.PT1M'+'.curated'+'.csv') #Ajoute le fichier csv dans data_curated
            return df2

    def prepare_chart_dict(self):  # Initialise le dictionnaire de clés resolution et de valeur dictionnaire ticker -> séries temporelles
        for resolution in ['PT1M','P1D']:
            for ticker in self.stocknames:
                df1=pd.read_csv('./data_brute/'+ticker+'.'+resolution+'.csv')
                if(resolution=='P1D'):
                    df2=pd.Series(list(df1['price']),index=list(pd.to_datetime(df1.iloc[:,0])))  # Création d'une serie temporelle
                else:
                    df2=self.curate(df1,ticker)                 # Filtre les données de résolution en minute
                self.chartdict[resolution][ticker]=df2

    ### Stratégie

    def buy_sell(self,total_avail,pos,logf):
        # total_avail: argent disponible dans le portefeuille
        # pos: nombre d"action dans le portefeuille
        # logf: fichier de log
        if (self.is_market_open()):
            date=self.now()
            offset = pd.DateOffset(minutes=30*60*24)
            start=date - offset
            end=date-pd.DateOffset(minutes=1) 
            df=self.chartdict['PT1M'][self.ticker]
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
            offset = pd.DateOffset(minutes=1)
            date = self.time - offset
            if self.time.day != date.day:
                while (not self.is_market_open()):  
                    self.sleep(60)
                p = self.get_latest_price(self.ticker)
                print("Total : %s" % (pooldict['pos']*p+pooldict['leftover']),end = " ") # Affiche le total des fonds (ie actifs + leftover)
            if(self.now()> self.end):
                print("Fin de la simulation ")
                return (100.0*(pooldict['total']+pooldict['leftover'])/initial_sum)  # Renvoie le ROI
            if (not self.is_market_open()):  # Si le marché est fermé on avance dans le temps d'une minute
                self.sleep(60)
                continue
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
                self.sleep(120)                 # On avance de 2 minutes
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
            self.sleep(60)  # On avance d'une minute
        logf.close()
        return (100*(pooldict['total']+pooldict['leftover'])/initial_sum)  # On renvoie le ROI en fin de simulation

### EXECUTION ###

if __name__ == "__main__":
    ticker1 = input("Veuillez entrer un Ticker : ")
    somme_ini = int(input("Veuillez entrer un montant initial : "))
    if not existe_fichier(ticker1):
        download()
    bot=Trading_bot(sim=True,ticker = ticker1)  # Création d'un bot pour la simulation
    bot.prepare_chart_dict()                    # Initialisation du dictionnaire des données
    print(bot.pool("Ordre_courant",somme_ini,False))

