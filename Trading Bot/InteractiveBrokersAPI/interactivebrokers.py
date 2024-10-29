from ib_insync import *
import pandas as pd
ib = IB()

# use this instead for IB Gateway
# ib.connect('127.0.0.1', 7497, clientId=1)

# us this for TWS (Workstation)
ib.connect('127.0.0.1', 7497, clientId=1)

stock = Stock('AMD', 'SMART', 'USD')

bars = ib.reqHistoricalData(
    stock, endDateTime='', durationStr='90 D',
    barSizeSetting='5 mins', whatToShow='MIDPOINT', useRTH=True)

# convert to pandas dataframe
df = util.df(bars)
df = df.dropna()
df.set_index('date', inplace=True)
df.index = df.index.tz_convert('Europe/Paris')
pd.set_option('display.max_rows', None)
#print(bars)
print(df.tail(10))

market_data = ib.reqMktData(stock, '', False, False)

def onPendingTicker(ticker):
    print("pending ticker event received")
    print(ticker)

ib.pendingTickersEvent += onPendingTicker

ib.run()