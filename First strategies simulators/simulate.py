# IMPORTATIONS
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time,datetime,pytz
import sys
import re
import time

def ponderation(x):
    return 2/np.pi*np.arctan(10*x)

def string_to_number(text, n=3):
    l = []
    for c in text:
        l.append('{:0>{}}'.format(ord(c), n))
    return int(''.join(l))

class trading_bot():
    def __init__(self,sim):

        # SETUP LOGGING
        logging.basicConfig(level=logging.INFO)
        self.debug=0
        self.plot=0
        self.logging=1
        self.verbose=1
        self.nr=0
        
        
        # SETUP SIM
        self.sim=sim
        self.time=pd.to_datetime('2022-09-15 15:30:00') # sim start
        self.end=datetime.datetime.now() # sim end

        # internal data #strategy Parameters
        self.stocknames=['AAPL']
        self.pool1="pool1.json"
        self.nasdaq_time_diff=6
        self.chartdict={
                'P1D': {},
                'PT1M': {}
            }
        self.trading_fee=2
        self.day_avg={}

        

    #def __del__(self):
    #    # LOGOUT
    #    print("Logout, session id : ", self.session_id)
    #    self.trading_api.logout()

    def buy_sell(self,total_avail,pos,logf):
        #-------------------------------------------------------------------------------#
        # a vous d'ajouter votre stratégie ici
        # total_avail: argent disponible dans le portefeuille
        # pos: nombre d"action dans le portefeuille
        # logf: fichier de log
        #-------------------------------------------------------------------------------#

        if (self.is_market_open('AAPL')):
            date=self.now()-pd.DateOffset(minutes=0)
            # past 30 days.
            offset = pd.DateOffset(minutes=30*60*24)
            start=date- offset
            end=date-pd.DateOffset(minutes=1) # we only know price upto last minute
            df=self.chartdict['PT1M']['AAPL']
            df_range=df[start:end]
            price=pd.Series(list(df_range[:]),index=range(0,len(df_range),1))
            m_mobc =  price.rolling(window=10).mean().iloc[-1]
            m_glic =  price.rolling(window=5).mean().iloc[-1]
            m_mobl = price.rolling(window=180).mean().iloc[-1]
            m_glil = price.rolling(window=20).mean().iloc[-1]
            taux_ct = m_glic/m_mobc
            taux_lt = m_glil/m_mobl
            if (taux_lt > 1.01 and taux_ct > 1.01) or (taux_lt > 1.01 and taux_ct < 0.9) :
                p=self.get_latest_price('AAPL')
                size=total_avail//p
                #print("  J'aimerais acheter "+str(size)+" actions a  " + str(p) +" $")
                return 'BUY','AAPL',p,size
            elif pos!=0 and ((taux_lt < 0.99 and taux_ct > 1.01) or (taux_lt < 0.99 and taux_ct < 0.98)) :
                p=self.get_latest_price('AAPL')
                size=pos
                #print("  J'aimerais vendre "+str(size)+" actions a  " + str(p) +" $")
                return 'SELL','AAPL',p,size
            else:
                return None,None,0,0 
        else:
            return None,None,0,0 

        #-------------------------------------------------------------------------------#
        # a vous d'ajouter votre stratégie ici
        #-------------------------------------------------------------------------------#
    def now(self):
        return self.time
    def sleep(self,secs):
        offset = pd.DateOffset(minutes=secs//60)    
        self.time=self.time+offset
    def log(self,string,logf):
        if(self.logging):
            print(string,file=logf)
    def get_latest_price(self,ticker):
        #df1=pd.read_csv(ticker+'.PT1M.csv')
        #df2=pd.Series(list(df1['price']),index=list(pd.to_datetime(df1.iloc[:,0])))
        df2=self.chartdict['PT1M'][ticker]
        if (self.time in df2.index):
            return df2[self.time]
        else:
            if(self.verbose):
                print(self.now(),"Market Closed")
            return False
    def is_dst(self,dt):
        timezone = pytz.timezone('America/New_York') #nasdaq nyse timezone
        aware_dt = timezone.localize(dt)
        return aware_dt.dst() != datetime.timedelta(0,0)
    def is_market_open(self,ticker):
        #df1=pd.read_csv(ticker+'.PT1M.csv')
        #df2=pd.Series(list(df1['price']),index=list(pd.to_datetime(df1.iloc[:,0])))
        if(not ticker): ticker='AAPL'
        df2=self.chartdict['PT1M'][self.stocknames[0]]
        if (self.time in df2.index):
            if(self.is_dst(self.time)):
                start=datetime.datetime(self.time.year,self.time.month,self.time.day,15,30)  
                if(self.time > start):
                    return True
                else:
                    return False
            else:
                start=datetime.datetime(self.time.year,self.time.month,self.time.day,14,30)
                if(self.time > start):
                    return True
                else:
                    return False
        else:
            return False
    def is_market_open_at(self,time):
        ticker='AAPL'
        df2=self.chartdict['PT1M'][self.stocknames[0]]
        if (time in df2.index):
            return True
        else:
            return False
    def is_closing_hour(self):
        if(self.sim):
            offset = pd.DateOffset(minutes=0)
            time=self.time+offset
            df2=self.chartdict['PT1M'][self.stocknames[0]]
            if (time in df2.index):
                return False
            else:
                return True
            
        
    def curate(self,df,ticker):
        errdate=None
        newdates=[]
        price=[]
        count=0
        for i in list(pd.to_datetime(df.iloc[:,0])):
            
            if(int(str(i.hour)) == 14 and int(str(i.minute)) == 30):
                errdate=[i.year,i.month,i.day ]
            elif(int(str(i.hour)) == 14 and int(str(i.minute)) == 31):
                errdate=[i.year,i.month,i.day ]
            if(errdate):
                if(errdate[0]==i.year and errdate[1]==i.month and errdate[2]==i.day):
                    newdate=i+pd.DateOffset(minutes=60)
                    newdates.append(newdate)
                    price.append(df.iloc[count,1])
                else:
                    #if(errdate[0]==i.year and errdate[1]==i.month and errdate[2]==i.day and i.hour==20 and i.minute==59):
                    errdate=None 
                    newdates.append(i)
                    price.append(df.iloc[count,1])
            else:
                #filtering times not expected. we expect  from 5:30-21:59
                if(int(str(i.hour)) == 15 and int(str(i.minute)) == 29):
                    pass
                    #print(i)
                elif(int(str(i.hour)) == 15 and int(str(i.minute)) == 28):
                    pass
                    #print(i)
                elif(int(str(i.hour)) == 22 and int(str(i.minute)) == 00):
                    pass
                    #print(i)
                else:
                    newdates.append(i)
                    price.append(df.iloc[count,1])
            #print(i,df1.iloc[:,1])
            #df[i]=df1.iloc[:,1] 
            count=count+1
        #price=list(df['price'])
        if(self.debug and ticker=='META'):
            print(newdates,price)
        i=0
        while (i < (len(newdates)-1)):
            #if(self.debug):
            #    print(len(newdates),len(price))
            if( newdates[i].year == newdates[i+1].year and
                newdates[i].month == newdates[i+1].month and
                newdates[i].day == newdates[i+1].day):
                diff=(newdates[i+1] -newdates[i]).total_seconds()//60
                diff=int(diff)
                if (diff!=1):
                    missing_prices=[]
                    missing_dates=[]
                    if(self.debug and ticker=='META'):
                        print('Err',diff,newdates[i],newdates[i+1],price[i],price[i+1])
                    for j  in  range(diff-1):
                        newdates.insert(i+j+1,newdates[i]+pd.DateOffset(minutes=j+1))
                        price.insert(i+j+1,price[i]+(j+1)*(price[i+1]-price[i])/diff)
                    i=i+diff
                else:
                    #print(newdates[i],newdates[i+1],price[i],price[i+1])
                    i=i+1
            else:
                #print(newdates[i],newdates[i+1],price[i],price[i+1])
                i=i+1
                    
            #print(i,df1.iloc[:,1])
            #df[i]=df1.iloc[:,1] 
        df2=pd.Series(price,index=newdates,name='price')
        #df2=pd.Series(list(df['price']),index=newdates,name='price')
        df2.to_csv(ticker+'.PT1M'+'.curated'+'.csv')
        return df2
    def prepare_chart_dict(self):
        for resolution in ['PT1M','P1D']:
            for ticker in self.stocknames:
                df1=pd.read_csv('./data/'+ticker+'.'+resolution+'.csv')
                if(resolution=='P1D'):
                    df2=pd.Series(list(df1['price']),index=list(pd.to_datetime(df1.iloc[:,0])))
                else:
                    df2=self.curate(df1,ticker)
                self.chartdict[resolution][ticker]=df2

        

    def pool(self,poolname,initial_sum,load_existing_pool=False):
        if(load_existing_pool):
            with open(poolname+'.json') as handle:  #using init file for the 1st time
                pooldict = json.loads(handle.read())
            initial_sum=pooldict['total']+pooldict['leftover']
        else:
            #----------------------------------------------------------------------
            #init pool
            #----------------------------------------------------------------------
            pooldict={}
            pooldict['active']=0
            pooldict['total']=initial_sum
            pooldict['total_exp']=initial_sum
            pooldict['orderid']=-1
            pooldict['ticker']=0
            pooldict['pos']=0
            pooldict['leftover']=0
            with open(poolname+'.json', 'w') as f:
                f.write(json.dumps(pooldict))
            #----------------------------------------------------------------------
        scanning=(pooldict['active']==0)
        pending=(pooldict['active']==1)
        logf=open(poolname+'.log', 'w')
        self.log("%s: start thread %s" % (self.now(),poolname),logf)
        logf.close()
        while(pending or scanning):
            logf=open(poolname+'.log', 'a+')
            ticker=pooldict['ticker']
            #-----------------------------------------------------------------------
            print("\r",self.now(), pooldict['total']+pooldict['leftover'],end="")
            if(self.now()> self.end):
                print("end of simulaiton\n")
                #sys.exit(0)
                return (100.0*(pooldict['total']+pooldict['leftover'])/initial_sum)
            if (not self.is_market_open(ticker)):
                self.sleep(60)
                continue
            #-----------------------------------------------------------------------
            if(pooldict['active']==2):
                latest_price=self.get_latest_price(ticker)
                if(latest_price): #market is open and data is not missing for that minute
                    if(latest_price > (pooldict['total_exp']/pooldict['pos'])):
                        pending,tot_trade=False, pooldict['total_exp']-2*self.trading_fee
                        self.log("%s: executed order %s,%s" % (self.now(),ticker ,tot_trade),logf)
                    else:
                        pending,tot_trade=True,pooldict['total_exp']
            else:
                pending,tot_trade=False,0.0
            #-----------------------------------------------------------------------
            #self.log("%s: pending order %s,%s" % (self.now(),pending,tot_trade),logf)
            #-----------------------------------------------------------------------
            if(pending):
                self.sleep(120)
                self.log(pooldict,logf)
            elif(not scanning and tot_trade!=0.0): # a trade done in last iter
                pooldict['active']=0
                pooldict['total']=tot_trade+pooldict['leftover']
                pooldict['total_exp']=tot_trade+pooldict['leftover']
                pooldict['orderid']=-1
                pooldict['ticker']=0
                pooldict['pos']=0
                pooldict['leftover']=0
                with open(poolname+'.json', 'w') as f:
                    f.write(json.dumps(pooldict))
                scanning=True
            else:
                #-----------------------------------------------
                total_avail=pooldict['total']
                pos=pooldict['pos']
                bs,ticker,price,size=self.buy_sell(total_avail,pos,logf)
                #self.log("%s: scanned trades: %s %f %f" % (self.now(),ticker,price,size),logf)
                if(pooldict['active']==0 and bs=="BUY"):
                    if(price*size < pooldict['total']):
                        buyorderid=self.now().isoformat()
                        print(" Ordre d'achat exécuté : " + str(size) + " à " + str(price) + " $" )
                    else:
                        buyorderid=0
                    if(not buyorderid): # funds insifficent e.g
                        logf.close()
                        continue  # restart scanning
                    buy_order_pending,tot_buy_trade=False,price*size
                    self.log("%s: Placed buy order  %s %s %f" % (self.now(),buyorderid,ticker,tot_buy_trade),logf)
                    total=pooldict['total'] #available sum
                    pooldict['total']=tot_buy_trade
                    pooldict['ticker']=ticker
                    pooldict['orderid']=buyorderid
                    pooldict['pos']=size
                    pooldict['active']=1
                    pooldict['leftover']=total-price*size
                    with open(poolname+'.json', 'w') as f:
                        f.write(json.dumps(pooldict))
                        if(self.verbose):
                            print(pooldict)
                elif(bs=="SELL"):
                    #calculating 1% over price but buy price can be diffretn (lower than price)
                    if(size <= pooldict['pos']):
                        sellorderid=self.now().isoformat()
                        print("ordre de vente éxécuté  : " + str(size) + " à " + str(price) + " $" )
                    if(sellorderid):
                        pooldict['active']=2
                        pooldict['total_exp']=price*size  #noting sel price in file 0.01*percent gives wrong results ????
                        pooldict['orderid']=sellorderid
                        with open(poolname+'.json', 'w') as f:
                            f.write(json.dumps(pooldict))
                            if(self.verbose):
                                print(pooldict)
                        scanning=False
                        pending=True
                        self.log("%s: Placed sale order  %s %s %f " % (self.now(),sellorderid,ticker,price*size),logf)
                    else:
                        self.log("Err: can not place sell order",logf)
                        scanning=False
                        pending=False   # exit the thread
                #else:
                #    self.log("Err: pooll is active, but no pending order found, something chanegd outside this thread",logf)
            self.sleep(60)
            #-----------------------------------------------------------------------
        logf.close()
        return (100*pooldict['total']/initial_sum)

    def run(self):
        #thread1=threading.Thread(target=strat.pool,args=("pool1"))
        thread2=threading.Thread(target=self.pool,args=(["pool2"]))
        #thread1.start()
        thread2.start()
        #thread1.join()
        thread2.join()
    def non_reg(self):
        # running sim for only todays data, with real pool compied to poolsim
        self.sim=True
        current=datetime.datetime.now() # sim start
        self.time=datetime.datetime(current.year,current.month,current.day,15,30)
        print("starting non-reg flow",self.time)
        self.prepare_chart_dict()
        print(self.pool("poolsim",0,True))

         
        
                
if __name__ == "__main__":
    strat=trading_bot(sim=True)
    strat.prepare_chart_dict()
    print(strat.pool("poolsim",5000,False))
