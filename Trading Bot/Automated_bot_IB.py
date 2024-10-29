### Automated Trading Bot Implemented with the set of strategies from the Strategy folder ###
import pandas as pd
import pandas_ta as ta
from apscheduler.schedulers.blocking import BlockingScheduler
from ib_insync import *
from backtesting import Strategy,Backtest
from datetime import datetime
from math import floor
import time
import threading
from DownloadData import download_data_bot
import importlib
# ============================================================================
### SRATEGIES AND MODELS ###
strategies = {1 : "ScalpingRSI_15min", 2 : "ScalpingBollinger_5min", 3 : "ScalpingLSTM_15min", 4 : "SwingLSTM_15min", 5 : "MomentumLSTM_5min" }

models = {1 : "NVDAmodel_15min_15_1", 2 : "AAPLmodel_15min_15_2", 3 : "SPYmodel_5min_15_2"}
# ============================================================================

def fitting_job(contract, timeframe,module,strat, durationStr = "30 D", end_date = ""):
    print("Fitting Job Started...")
    dfstream = download_data_bot(contract, timeframe, durationStr, end_date)
    if hasattr(module, 'apply_save'):
        module.apply_save("current_fitting",backtest = True)
    else:
        print(f"The module {strat.__name__} hasn't any function 'apply_save'.")
    dfstream = pd.read_csv("current_fitting.csv")
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
                sl1 = self.data.close[-1] + slatrls
                tp1 = self.data.close[-1] - slatr*TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(dfstream, strat, cash=250, margin=1/30)
    stats, heatmap = bt.optimize(slcoef=[i/5 for i in range(5, 26)],
                        TPSLRatio=[i/5 for i in range(5, 26)],
                        maximize='Return [%]', max_tries=500,
                        random_state=0,
                        return_heatmap=True)
    print(stats)
    slcoef = stats["_strategy"].slcoef
    TPSLRatio = stats["_strategy"].TPSLRatio
    with open("fitting_data_file.txt", "a") as file:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"SLcoef: {slatrcoef}, TPSLRatio: {TPSLRatio_coef}, expected return: {stats['Return [%]']}\n")
        file.write(f"Date: {date}\n")
    print("Fitting Job Finished...")
    return slcoef, TPSLRatio
    

    
def trading_job(contract, timeframe, durationStr, module,strat,slcoef = 1.5, TPSLRatio = 1.5,max_spread = 1,cash = cash,end_date =""):
    print("Trading Job Started...")
    dfstream = download_data_bot(contract, timeframe, durationStr, end_date)
    if hasattr(module, 'apply_save'):
        module.apply_save("current_bot",backtest = True)
    else:
        print(f"The module {strat.__name__} hasn't any function 'apply_save'.")
    dfstream = pd.read_csv("current_bot.csv")
    signal = dfstream.TotalSignal[-1]
    lotsize = cash//dfstream.close.iloc[-1]
    now = datetime.now()
    if now.weekday() == 0 and now.hour < 15 and now.minute < 5:  # Monday before 15:05
        slcoef, TPSLRatio = fitting_job(contract, timeframe, module,strat)
        print(f"slcoef : {slcoef}, TPSLRatio : {TPSLRatio}")
    slatr = slcoef*dfstream.ATR.iloc[-1]

    ticker = ib.ticker(contract)
    candle_open_bid = ticker.bid
    candle_open_ask = ticker.ask
    spread = candle_open_ask-candle_open_bid

    SLBuy = candle_open_bid-slatr-spread
    SLSell = candle_open_ask+slatr+spread

    TPBuy = candle_open_ask+slatr*TPSLRatio+spread
    TPSell = candle_open_bid-slatr*TPSLRatio-spread

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

def launch_bot():
    strategy = input("Enter a strategy to backtest from the Strategy folder:\n 1 for ScalpingRSI_15min\n 2 for ScalpingBollinger_5min\n 3 for ScalpingLSTM_15min\n 4 for SwingLSTM_15min\n 5 for MomentumLSTM_5min\n")
    if not(os.path.exists("Strategy/"+strategies[int(strategy)]+".py")):
        print("The strategy does not exist.")
        return
    module = importlib.import_module("Strategy."+strategies[int(strategy)])
    if int(strategy) in [3,4,5]:
        model = input("Enter the model to use for the strategy:\n (Scalping <- 1 indicator | Swing,Momentum <- 2 indicators)\n 1 for NVDAmodel_15min_15_1\n 2 for AAPLmodel_15min_15_2\n 3 for SPYmodel_5min_15_2\n")
        if hasattr(module,'set_model_name'):
            module.set_model_name(models[int(model)])
            module.LoadModel()
    secType = input("Enter the security type (STK = Stock (or ETF), OPT = Option, FUT = Future, IND = Index, FOP = Futures option, CASH = Forex pair, CFD = CFD, BAG = Combo, WAR = Warrant, BOND = Bond, CMDTY = Commodity, NEWS = News, FUND = Mutual fund, CRYPTO = Crypto currency, EVENT = Bet on an event): ")
    ticker_symbol = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (‘1 secs’, ‘5 secs’, ‘10 secs’ 15 secs’, ‘30 secs’, ‘1 min’, ‘2 mins’, ‘3 mins’, ‘5 mins’, ‘10 mins’, ‘15 mins’, ‘20 mins’, ‘30 mins’, ‘1 hour’, ‘2 hours’, ‘3 hours’, ‘4 hours’, ‘8 hours’, ‘1 day’, ‘1 week’, ‘1 month’): ")
    durationStr = input("Enter the duration ( ‘60 S’, ‘30 D’, ‘13 W’, ‘6 M’, ‘10 Y’.): ")
    end_date = input("Enter the end date (‘yyyyMMdd HH:mm:ss’ or enter for now): ")
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=13)
    contract = Contract()
    contract.symbol = ticker_symbol
    contract.secType = secType
    contract.exchange = "SMART"
    contract.currency = "USD"                                                                                                                                                                                                                                                                                                                    
    if hasattr(module,strategies[int(strategy)]):
        strat = getattr(module,strategies[int(strategy)])
    slcoef = strat.slcoef; TPSLRatio = strat.TPSLRatio
    cash = input("Enter the cash amount that will be used by the strategy: ")
    max_spread = input("Enter the maximum spread allowed for the strategy: ")
    def trade():
        trading_job(contract, timeframe, durationStr, module,strat, slcoef=slcoef, TPSLRatio=TPSLRatio, max_spread=max_spread)
    scheduler = BlockingScheduler()
    if timeframe == "5 mins":
        min = '1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56'; misfire = 15
    elif timeframe == "15 mins":
        min = '1, 16, 31, 46'; misfire = 30
    scheduler.add_job(trade, 'cron', day_of_week='mon-fri', hour='15-22', minute=min, timezone='Europe/Paris', misfire_grace_time=misfire)
    scheduler.add_job(stop_job, run_date=datetime.now() + timedelta(days=30))
    scheduler.start()

launch_bot()