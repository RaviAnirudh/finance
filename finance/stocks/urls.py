
from django.urls import path
from .views import backtest_view, fetch_stocks_view, predict_stock_view, report_view

urlpatterns = [
    path('fetch-stocks', fetch_stocks_view, name='fetch_stock'),
    path('backtest', backtest_view, name='backtest'),
    path('predict-stock', predict_stock_view, name='predict_stock'),
    path('report/', report_view, name='report'),
]