import yfinance as yf
import pandas as pd
import os
from ib_insync import Stock,IB, util, Contract
import pandas as pd
import datetime

def download_data_yfinance():
    ticker_symbol = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    ticker = yf.Ticker(ticker_symbol)

    df = ticker.history(start=start_date, end=end_date, interval=timeframe)
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    df.index = df.index.tz_convert('Europe/Paris')
    df.to_csv(f"Strategy/Data/{ticker_symbol.upper()}_{timeframe.upper()}_{start_date}_{end_date}.csv")

    print("Data downloaded successfully in Data folder of Strategy.")


def download_data_ib(backtest = False):
    secType = input("Enter the security type (STK = Stock (or ETF), OPT = Option, FUT = Future, IND = Index, FOP = Futures option, CASH = Forex pair, CFD = CFD, BAG = Combo, WAR = Warrant, BOND = Bond, CMDTY = Commodity, NEWS = News, FUND = Mutual fund, CRYPTO = Crypto currency, EVENT = Bet on an event): ")
    ticker_symbol = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (‘1 secs’, ‘5 secs’, ‘10 secs’ 15 secs’, ‘30 secs’, ‘1 min’, ‘2 mins’, ‘3 mins’, ‘5 mins’, ‘10 mins’, ‘15 mins’, ‘20 mins’, ‘30 mins’, ‘1 hour’, ‘2 hours’, ‘3 hours’, ‘4 hours’, ‘8 hours’, ‘1 day’, ‘1 week’, ‘1 month’): ")
    durationStr = input("Enter the duration ( ‘60 S’, ‘30 D’, ‘13 W’, ‘6 M’, ‘10 Y’.): ")
    end_date = input("Enter the end date (‘yyyyMMdd HH:mm:ss’ or enter for now): ")
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)
    contract = Contract()
    contract.symbol = ticker_symbol
    contract.secType = secType
    contract.exchange = "SMART"
    contract.currency = "USD"
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date,
        durationStr=durationStr,
        barSizeSetting=timeframe,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1)
    ib.disconnect()
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for i in range(len(bars)):
        df.loc[i,'Date'] = bars[i].date
        df.loc[i,'Open'] = bars[i].open
        df.loc[i,'High'] = bars[i].high
        df.loc[i,'Low'] = bars[i].low
        df.loc[i,'Close'] = bars[i].close
        df.loc[i,'Volume'] = bars[i].volume
    df.set_index('Date', inplace=True)
    if timeframe in ["1 secs", "5 secs", "10 secs", "15 secs", "30 secs", "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "20 mins", "30 mins", "1 hour", "2 hours", "3 hours", "4 hours", "8 hours"]:
        df.index = df.index.tz_convert('Europe/Paris')
    timeframe = timeframe.replace(" ", "")
    durationStr = durationStr.replace(" ", "")
    if end_date == "":
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = end_date[:4]+"-"+end_date[4:6]+"-"+end_date[6:8]
    if os.path.exists(f"Strategy/Data/{ticker_symbol.upper()}_{timeframe.upper()}_{durationStr}_{end_date}.csv") and not backtest:
        print("File already exists. Download stopped.")
        return f"{ticker_symbol.upper()}_{timeframe.upper()}_{durationStr}_{end_date}"
    df.to_csv(f"Strategy/Data/{ticker_symbol.upper()}_{timeframe.upper()}_{durationStr}_{end_date}.csv")
    name = f"{ticker_symbol.upper()}_{timeframe.upper()}_{durationStr}_{end_date}"
    print("Data downloaded successfully in Data folder of Strategy.")
    return name

if input("Enter 1 for yfinance (less possibilities) and 2 for Interactive Brokers: ") == "1":
    download_data_yfinance()
else:
    download_data_ib()


### USABLE ONLY WHILE CONNECTED WITH THE IB API ###

def download_data_bot(contract, timeframe, durationStr, end_date =""):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date,
        durationStr=durationStr,
        barSizeSetting=timeframe,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1)
    df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for i in range(len(bars)):
        df.loc[i,'Date'] = bars[i].date
        df.loc[i,'Open'] = bars[i].open
        df.loc[i,'High'] = bars[i].high
        df.loc[i,'Low'] = bars[i].low
        df.loc[i,'Close'] = bars[i].close
        df.loc[i,'Volume'] = bars[i].volume
    df.set_index('Date', inplace=True)
    df.index = df.index.tz_convert('Europe/Paris') # Change the timezone to Paris because the timeframe is smaller than a day
    return df