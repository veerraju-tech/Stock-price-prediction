#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

import yfinance as yf

# Define the ticker symbol
ticker_symbol = input("Enter Ticker Symbol: ")

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical market data
historical_data = ticker.history(period="max")

# Load your data (replace 'your_stock_data.csv' with your file)
historical_data.to_csv('company_data.csv', header=True, index=True)

# Load your data (replace 'your_stock_data.csv' with your file)
data = pd.read_csv('company_data.csv')

data['Date'] = pd.to_datetime(data['Date'], utc=True)
# Ensure the data contains a 'Close' column for the stock's closing prices
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Function to create sequences of data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Hyperparameters
sequence_length = 60

# Prepare the data for the LSTM model
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape input data to [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    Input(shape=(X_train.shape[1],1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Invert scaling for actual prices
actual_prices = scaler.inverse_transform(y_test)

# Print the next day's predicted price (the last predicted value)
next_day_predicted_price = predicted_prices[-1][0]


# In[2]:


train_dates = data['Date'][:train_size]
test_dates = data['Date'][train_size + sequence_length:]

next_day_date = test_dates.iloc[-1] + pd.Timedelta(days=1)


# In[3]:


# Plot the results
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(test_dates, actual_prices, label='Actual Prices', color='green')

# Plot predicted prices
plt.plot(test_dates, predicted_prices, label='Predicted Prices', color='red')

# Highlight the next day's predicted price as a dot
plt.scatter(next_day_date, next_day_predicted_price, label="Next Day's Predicted Price", color='black', s=25, marker='o')

# Add labels, title, and legend
plt.title(f'Stock Price Prediction for {ticker_symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the next day's predicted price and date
print(f"Next Day's Predicted Price ({next_day_date.date()}): {next_day_predicted_price:.2f}")


# In[4]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Calculate MSE, MAE, and RMSE
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)

# Print the accuracy metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[5]:


data.tail()


# In[ ]:




