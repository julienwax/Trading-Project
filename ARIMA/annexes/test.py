import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load your stock price data into a DataFrame
# Replace 'your_data.csv' with the path to your CSV file
def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns = {'Unnamed: 0': 'Date'})
    # Assuming your CSV file has columns 'Date' and 'Price'
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Function to train ARIMA model and make predictions
def fit_arima_model(data, order):
    # Fit ARIMA model
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Function to predict the stock price for the next day
def predict_next_day_price(model_fit, date):
    # Predict the next day price
    next_day = date + pd.Timedelta(days=1)
    prediction = model_fit.forecast(steps=1)[0]  # Forecast the next day
    return prediction

# Example usage:
# Load your stock price data
file_path = 'data_brute/BTC-EUR.P1D.csv'  # Replace with your file path
stock_data = load_stock_data(file_path)

# Define the ARIMA model order (p, d, q)
order = (1, 0, 1)  # Example order, you can adjust these parameters

# Fit the ARIMA model
model_fit = fit_arima_model(stock_data, order)

# Provide a specific date for prediction
prediction_date = pd.to_datetime('2024-02-20')  # Example date, change as needed

# Predict the stock price for the next day
next_day_prediction = predict_next_day_price(model_fit, prediction_date)
print("Predicted stock price for", prediction_date.date(), ":", next_day_prediction)
