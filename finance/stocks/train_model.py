import os
import sys
import django
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finance.settings')
django.setup()

from finance.stocks.models import StockPrice 

def train_and_save_model(symbol='AAPL'):
    stock_data = StockPrice.objects.filter(symbol=symbol).order_by('date')
    if not stock_data.exists():
        print("No data found for symbol:", symbol)
        return
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['close_price'] = df['close_price'].astype(float)

    X = df.index.values.reshape(-1, 1)  
    y = df['close_price'].values 

    model = LinearRegression()
    model.fit(X, y)
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
train_and_save_model('AAPL')
