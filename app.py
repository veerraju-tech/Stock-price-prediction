from flask import Flask, render_template, request
import os
import time
import io
import base64
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yfinance as yf

app = Flask(__name__)

# Load the pre-trained model
model = load_model("stock_model.h5")

# Cache the scaler for consistency
scaler = MinMaxScaler(feature_range=(0, 1))


@app.route("/amazon_stock")
def amazon_stock():
    ticker_symbol = "AMZN"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))
        
@app.route("/tcs_stock")
def tcs_stock():
    ticker_symbol = "TCS.NS"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route("/tesla_stock")
def tesla_stock():
    ticker_symbol = "TSLA"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route("/microsoft_stock")
def microsoft_stock():
    ticker_symbol = "MSFT"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route("/apple_stock")
def apple_stock():
    ticker_symbol = "AAPL"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route("/bitcoin_stock")
def bitcoin_stock():
    ticker_symbol = "BTC-USD"  # Amazon's stock symbol
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    

# Function to fetch and cache stock data
def fetch_stock_data(ticker_symbol):
    cache_file = f"{ticker_symbol}_data.csv"
    if os.path.exists(cache_file):
        # Use cached data if it's recent (e.g., last 24 hours)
        if time.time() - os.path.getmtime(cache_file) < 86400:  # 24 hours
            return pd.read_csv(cache_file)
    # Fetch new data and cache it
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="max")
    historical_data.to_csv(cache_file)
    return pd.read_csv(cache_file)

# Function to process the stock ticker
def process_stock(ticker_symbol):
    # Fetch and preprocess data
    data = fetch_stock_data(ticker_symbol)
    data['Date'] = pd.to_datetime(data['Date'])
    closing_prices = data['Close'].values.reshape(-1, 1)

    # Normalize the data using the cached scaler
    global scaler
    scaled_data = scaler.fit_transform(closing_prices)

    sequence_length = 60
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    # Create sequences for prediction
    test_size = int(len(scaled_data) * 0.2)
    test_data = scaled_data[-(test_size + sequence_length):]  # Use only the most recent data
    X_test, y_test = create_sequences(test_data, sequence_length)

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Invert scaling for actual prices
    y_test = scaler.inverse_transform(y_test)

    # Predict the next day's price
    next_day_predicted_price = predicted_prices[-1][0]

    # Get dates for plotting
    test_dates = data['Date'][-len(y_test):]
    next_day_date = test_dates.iloc[-1] + pd.Timedelta(days=1)

    # Calculate accuracy metrics
    mse = mean_squared_error(y_test, predicted_prices)
    mae = mean_absolute_error(y_test, predicted_prices)
    rmse = np.sqrt(mse)

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual Prices', color='green')
    plt.plot(test_dates, predicted_prices, label='Predicted Prices', color='red')
    plt.scatter(next_day_date, next_day_predicted_price, label="Next Day's Predicted Price", color='black', s=25, marker='o')
    plt.title(f'Stock Price Prediction for {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()

    # Convert plot to Base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return graph_url, next_day_predicted_price, mse, mae, rmse

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict.html')
def predict():
    return render_template('predict.html')
    
@app.route('/services.html')
def services():
    return render_template('services.html')
    
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/article.html')
def article():
    return render_template('article.html')
    
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/result', methods=['POST'])
def result():
    ticker_symbol = request.form['ticker_symbol']
    try:
        graph_url, next_day_price, mse, mae, rmse = process_stock(ticker_symbol)
        return render_template(
            'result.html',
            ticker_symbol=ticker_symbol,
            graph_url=graph_url,
            next_day_price=next_day_price,
            mse=mse,
            mae=mae,
            rmse=rmse
        )
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
