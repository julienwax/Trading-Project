import ibapi
from ib_insync import IB, Stock, MarketOrder

# Connect to TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Use your correct port and clientId

# Define the contract and order details
contract = Stock('AAPL', 'SMART', 'USD')
order = MarketOrder('BUY', 1)  # Buy 1 share of AAPL at market price

# Place the order
trade = ib.placeOrder(contract, order)

# Wait for the order to be filled
ib.waitOnUpdate()
print(trade)

# Disconnect
ib.disconnect()


