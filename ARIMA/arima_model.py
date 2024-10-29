import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns = {'Unnamed: 0': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df.set_index('Date', inplace=True)
    df.index.freq = 'D'
    return df

def fit_arima_model(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

def predict_next_day_price(model_fit, date):
    next_day = date + pd.Timedelta(days=1)
    prediction = model_fit.forecast(steps=1).iloc[0] 
    return prediction

file_path = 'data_brute/BTC-EUR.P1D.csv'
stock_data = load_stock_data(file_path)

# Define the ARIMA model order (p, d, q)
order = (2, 1, 2) 

def prediction(date,df1):
    df = df1.copy()
    df = df.rename(columns = {'Unnamed: 0': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df.set_index('Date', inplace=True)
    df.index.freq = 'D'
    df2 = df.loc[:date]
    #auto_model= auto_arima(df2, seasonal=False, stepwise=True, suppress_warnings=True)
    #order = auto_model.order
    model = ARIMA(df2, order=order)
    model_fit = model.fit()
    return predict_next_day_price(model_fit,date)

#print(prediction(pd.to_datetime('2023-06-20'),df1 = pd.read_csv('data_brute/BTC-EUR.P1D.csv')))