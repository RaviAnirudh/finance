import base64
from decimal import Decimal
from io import BytesIO
from django.http import JsonResponse
import matplotlib
import pdfkit
import requests
import pandas as pd
from .models import StockPrediction, StockPrice
from django.utils.dateparse import parse_date
import numpy as np
import pickle
from datetime import timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from matplotlib import pyplot as plt
from io import BytesIO
import base64
import pdfkit 
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

matplotlib.use('Agg') 
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
url = os.getenv('ALPHA_VANTAGE_URL')

def fetch_stocks_data(symbol):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()
    if not isinstance(data, dict):
        return JsonResponse({'status': 'error', 'message': 'Invalid response from API, expected dict but got list or other type'})

    if 'Time Series (Daily)' not in data:
        return JsonResponse({'status': 'error', 'message': 'Error fetching data from Alpha Vantage'})

    daily_data = data['Time Series (Daily)']

    daily_data_list = list(daily_data.items())[:2000]

    stock_prices = []
    for date_str, price_data in daily_data_list:
        date = parse_date(date_str)
        stock_prices.append(StockPrice(
            symbol=symbol,
            date=date,
            open_price=price_data['1. open'],
            high_price=price_data['2. high'],
            low_price=price_data['3. low'],
            close_price=price_data['4. close'],
            volume=price_data['5. volume']
        ))

    StockPrice.objects.bulk_create(stock_prices, ignore_conflicts=True)

    return JsonResponse({'status': 'success', 'message': f"Inserted {len(stock_prices)} records for symbol {symbol}"})

    
def calculate_moving_averages(symbol, short_window=50, long_window=200):
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')

    if not stock_data.exists():
        raise ValueError(f"No data found for symbol {symbol}")
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['close_price'] = df['close_price'].astype(float)
    df.set_index('date', inplace=True)

    df['short_mavg'] = df['close_price'].rolling(window=short_window, min_periods=1).mean()
    df['long_mavg'] = df['close_price'].rolling(window=long_window, min_periods=1).mean()

    return df

def backtest_strategy(symbol, initial_investment, short_window=50, long_window=200):
    df = calculate_moving_averages(symbol, short_window, long_window)

    cash = initial_investment 
    holdings = 0  
    portfolio_value = initial_investment
    trades = [] 

    for i in range(1, len(df)):
        current_price = df['close_price'].iloc[i]
        previous_short_mavg = df['short_mavg'].iloc[i - 1]
        previous_long_mavg = df['long_mavg'].iloc[i - 1]

        if previous_short_mavg < previous_long_mavg and df['short_mavg'].iloc[i] > df['long_mavg'].iloc[i]:
            shares_to_buy = cash // current_price
            cash -= shares_to_buy * current_price
            holdings += shares_to_buy
            trades.append({'date': df.index[i], 'type': 'buy', 'price': current_price, 'shares': shares_to_buy})

        elif previous_short_mavg > previous_long_mavg and df['short_mavg'].iloc[i] < df['long_mavg'].iloc[i]:
            cash += holdings * current_price
            trades.append({'date': df.index[i], 'type': 'sell', 'price': current_price, 'shares': holdings})
            holdings = 0
        portfolio_value = cash + holdings * current_price

    total_return = (portfolio_value - initial_investment) / initial_investment * 100

    df['portfolio_value'] = df['close_price'] * holdings + cash
    max_drawdown = ((df['portfolio_value'].cummax() - df['portfolio_value']) / df['portfolio_value'].cummax()).max() * 100

    return {
        'initial_investment': initial_investment,
        'final_portfolio_value': portfolio_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'trades': trades
    }


MODEL_PATH = 'linear_regression_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_stock_prices(symbol, days=30):
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
    if not stock_data.exists():
        raise ValueError(f"No data found for symbol {symbol}")

    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['close_price'] = df['close_price'].astype(float)

    df['date'] = pd.to_datetime(df['date'])

    model = load_model()

    last_day = df['date'].max()
    future_dates = [last_day + timedelta(days=i) for i in range(1, days + 1)]
    future_days = np.array(range(len(df), len(df) + days)).reshape(-1, 1)

    predicted_prices = model.predict(future_days)

    predictions = []
    for i, date in enumerate(future_dates):
        price = predicted_prices[i]
        StockPrediction.objects.update_or_create(
            symbol=symbol, date=date, defaults={'predicted_price': price}
        )
        predictions.append({'date': date, 'predicted_price': price})

    return predictions

def generate_backtest_report(backtest_result):
    report = {
        'initial_investment': backtest_result['initial_investment'],
        'final_portfolio_value': backtest_result['final_portfolio_value'],
        'total_return': f"{backtest_result['total_return']:.2f}%",
        'max_drawdown': f"{backtest_result['max_drawdown']:.2f}%",
        'number_of_trades': len(backtest_result['trades']),
        'trades': backtest_result['trades']
    }
    return report

def generate_prediction_report(symbol, predictions):
    report = {
        'symbol': symbol,
        'predictions': predictions
    }
    return report


def plot_price_predictions(actual_prices, predicted_prices, future_dates):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices['date'], actual_prices['close_price'], label='Actual Prices', color='blue')
    plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='green', linestyle='--')

    plt.title('Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

def generate_pdf_report(report_data, plot_image_base64=None):
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica", 16)
    pdf.drawString(100, 750, "Stock Report")

    pdf.setFont("Helvetica", 12)
    y = 700 
    
    for key, value in report_data.items():
        if key != 'price_chart':  
            pdf.drawString(100, y, f"{key}: {value}")
            y -= 20 
    if plot_image_base64:
        image_data = base64.b64decode(plot_image_base64)
        image = ImageReader(BytesIO(image_data))

        pdf.drawImage(image, 100, y-300, width=400, height=300)
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
