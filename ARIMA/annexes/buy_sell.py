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

### Stratégie initiale:

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

###  STATEGIE ARIMA

def prediction():
    return None

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
            if prediction(self.time) > 1.01*prediction(self.time-pd.DateOffset(minutes=1)):
                p=self.get_latest_price(self.ticker)
                size=total_avail//p
                return 'BUY',self.ticker,p,size
            elif prediction(self.time) < 0.99*prediction(self.time-pd.DateOffset(minutes=1)):
                p=self.get_latest_price(self.ticker)
                size=pos
                return 'SELL',self.ticker,p,size
            else:
                return None,None,0,0 
        else:
            return None,None,0,0 