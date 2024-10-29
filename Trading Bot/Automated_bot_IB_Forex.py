### Automated Trading Bot Implemented on the Scalping Bollinger Strategy ###
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.blocking import BlockingScheduler
from ib_insync import *
from backtesting import Strategy,Backtest
from datetime import datetime
from math import floor
import time
import threading
# ============================================================================
### OPENED AND CLOSED ORDERS DONT EXIST ON TWS THEREFOR THE COUNT_OPENED_TRADES FUNCTION CONSIDER A LONG IS PLACED IS EUR
### EQUITY IS OVER 5000 AND A SHORT IS PLACED IF EUR EQUITY IS UNDER 4000

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=13)
contract = Forex('EURUSD')
ib.qualifyContracts(contract)
ib.reqMktData(contract, '', False, False)


lotsize = 1000

# SIGNALS

def ema_signal(df, current_candle, backcandles):
    start = max(0, current_candle - backcandles)
    end = current_candle
    relevant_rows = df.iloc[start:end]
    if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
        return 1
    elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
        return 2
    else:
        return 0
    
def total_signal(df, current_candle, backcandles):
    if (ema_signal(df, current_candle, backcandles)==2
        and df.close[current_candle]<=df['BBL_15_1.5'][current_candle]
        #and df.RSI[current_candle]<60
        ):
            return 2
    if (ema_signal(df, current_candle, backcandles)==1
        and df.close[current_candle]>=df['BBU_15_1.5'][current_candle]
        #and df.RSI[current_candle]>40
        ):
    
            return 1
    return 0

# ============================================================================

def get_bars(durationStr,barSizeSetting):
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr=durationStr,barSizeSetting=barSizeSetting, whatToShow='MIDPOINT', useRTH=True, formatDate=1)
    return bars

def count_opened_trades():
    account_summary = ib.accountSummary()
    eur_balance = None
    for item in account_summary:
        if item.tag == 'CashBalance' and item.currency == 'EUR':
            eur_balance = float(item.value)
    if eur_balance > 5000 or eur_balance < 4000:
        return 1
    else:
        return 0

def get_candles_frame(durationStr,barSizeSetting):
    bars = get_bars(durationStr,barSizeSetting)
    dfstream = pd.DataFrame(columns=['date','open','high','low','close'])
    for i in range(len(bars)):
        dfstream.loc[i,'date'] = bars[i].date
        dfstream.loc[i,'open'] = bars[i].open
        dfstream.loc[i,'high'] = bars[i].high
        dfstream.loc[i,'low'] = bars[i].low
        dfstream.loc[i,'close'] = bars[i].close
    dfstream["ATR"] = ta.atr(dfstream.high, dfstream.low, dfstream.close, length=7)
    dfstream["EMA_fast"]=ta.ema(dfstream.close, length=30)
    dfstream["EMA_slow"]=ta.ema(dfstream.close, length=50)
    dfstream['RSI']=ta.rsi(dfstream.close, length=10)
    my_bbands = ta.bbands(dfstream.close, length=15, std=1.5)
    dfstream=dfstream.join(my_bbands) 
    dfstream['TotalSignal'] = dfstream.apply(lambda row: total_signal(dfstream, row.name, 7), axis=1)
    dfstream.set_index('date', inplace=True)
    dfstream.index = dfstream.index.tz_convert('Europe/Paris')
    return dfstream

pd.set_option('display.max_rows', None)
#print(get_candles_frame("1 W", "5 mins").head(10))

slatrcoef = 1.1
TPSLRatio_coef = 1.5

def fitting_job():
    global slatrcoef
    global TPSLRatio_coef
    
    dfstream = get_candles_frame('1 W','5 mins')
    
    def SIGNAL():
        return dfstream.TotalSignal

    class MyStrat(Strategy):
        mysize = 3000 
        slcoef = 1.1
        TPSLRatio = 1.5
        
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            slatr = self.slcoef*self.data.ATR[-1]
            TPSLRatio = self.TPSLRatio
           
            if self.signal1==2 and len(self.trades)==0:
                sl1 = self.data.close[-1] - slatr
                tp1 = self.data.close[-1] + slatr*TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
            
            elif self.signal1==1 and len(self.trades)==0:         
                sl1 = self.data.close[-1] + slatr
                tp1 = self.data.close[-1] - slatr*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(dfstream, MyStrat, cash=250, margin=1/30)
    stats, heatmap = bt.optimize(slcoef=[i/10 for i in range(10, 26)],
                        TPSLRatio=[i/10 for i in range(10, 26)],
                        maximize='Return [%]', max_tries=300,
                        random_state=0,
                        return_heatmap=True)
    print(stats)
    slatrcoef = stats["_strategy"].slcoef
    TPSLRatio_coef = stats["_strategy"].TPSLRatio

    #with open("fitting_data_file.txt", "a") as file:
       #file.write(f"SLcoef: {slatrcoef}, TPSLRatio: {TPSLRatio_coef}, expected return: {stats['Return [%]']}\n")

    
def trading_job():
    print("Trading Job Started...")
    dfstream = get_candles_frame('1 D','5 mins')
    signal = total_signal(dfstream, len(dfstream)-1, 7) # current candle looking for open price entry
    
    global slatrcoef
    global TPSLRatio_coef    
    
    now = datetime.now()
    if now.weekday() == 0 and now.hour < 7 and now.minute < 5:  # Monday before 07:05
        fitting_job()
        print(slatrcoef, TPSLRatio_coef)

    slatr = slatrcoef*dfstream.ATR.iloc[-1]
    TPSLRatio = TPSLRatio_coef
    max_spread = 16e-5

    ticker = ib.ticker(contract)
    candle_open_bid = ticker.bid
    candle_open_ask = ticker.ask
    spread = candle_open_ask-candle_open_bid

    decimal = 4
    SLBuy = candle_open_bid-slatr-spread
    SLSell = candle_open_ask+slatr+spread
    SLBuy = floor((SLBuy) * 10**decimal) / (10**decimal)
    SLSell = floor(SLSell * 10**decimal) / (10**decimal)

    TPBuy = candle_open_ask+slatr*TPSLRatio+spread
    TPSell = candle_open_bid-slatr*TPSLRatio-spread
    TPBuy = floor(TPBuy * 10**decimal) / (10**decimal)
    TPSell = floor(TPSell * 10**decimal) / (10**decimal)

    if signal == 1 and count_opened_trades() == 0 and spread<max_spread:
        print("Sell Signal Found...")
        print("SLBuy: ", SLBuy, "SLSell: ", SLSell, "TPBuy: ", TPBuy, "TPSell: ", TPSell)
        order = MarketOrder('SELL', lotsize)
        stopLoss = StopOrder('BUY',lotsize, SLSell)
        takeProfit = LimitOrder('BUY', lotsize, TPSell)
        ib.placeOrder(contract, order)
        ib.placeOrder(contract, stopLoss)
        ib.placeOrder(contract, takeProfit)
        print("Sell Order Placed.")

    elif signal == 2 and count_opened_trades() == 0 and spread<max_spread:
        print("Buy Signal Found...")
        order = MarketOrder('BUY', lotsize)
        stopLoss = StopOrder('SELL',lotsize, SLBuy)
        takeProfit = LimitOrder('SELL', lotsize, TPBuy)
        ib.placeOrder(contract, order)
        ib.placeOrder(contract, stopLoss)
        ib.placeOrder(contract, takeProfit)
        print("Buy Order Placed.")
    print("Trading Job Finished...")

def stop_job():
    ib.disconnect()
    threading.Thread(target=scheduler.shutdown).start()

"""
scheduler = BlockingScheduler()
scheduler.add_job(trading_job, 'cron', day_of_week='mon-fri', hour='15-22', minute='1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56', timezone='Europe/Paris', misfire_grace_time=15)
scheduler.add_job(stop_job, run_date=datetime.now() + timedelta(minutes=30))
scheduler.start()
"""

trading_job()
ib.disconnect()
