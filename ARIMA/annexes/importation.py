
### IMPORTATIONS

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

symb = 'TSLA'
data = yf.Ticker(symb)
data_price = data.history(period = "max", interval = "1m" )
data_price1 = data.history(period = "max", interval = "1d")
#data_price = data_price.drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
#data_price1 = data_price1.drop(['Open','High','Low','Adj Close','Volume'], axis = 1)
data_price.to_csv("./data_brute/"+symb+".PT1M.csv")
data_price1.to_csv("./data_brute/"+symb+".P1D.csv")

