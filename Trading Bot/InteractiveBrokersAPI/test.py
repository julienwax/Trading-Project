from ib_insync import *
util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Stock('AMD', 'SMART', 'USD')

#ib.reqHeadTimeStamp(contract, whatToShow='TRADES', useRTH=True)
bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='3 Y',
        barSizeSetting='1 hour',
        whatToShow='TRADES',
        useRTH=True)

for i in range(len(bars)):
    if i % 100 == 0:
        print(bars[i].volume)
