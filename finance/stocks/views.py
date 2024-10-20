
from django.http import HttpResponse, JsonResponse
import pandas as pd

from finance.stocks.models import StockPrice
from .utils import backtest_strategy, fetch_stocks_data, generate_backtest_report, generate_pdf_report, generate_prediction_report, plot_price_predictions, predict_stock_prices
from rest_framework.decorators import api_view

@api_view(['GET'])
def fetch_stocks_view(request):
    symbol = request.GET.get('symbol', 'AAPL')  
    print("THis is symbol ",symbol)
    try:
        fetch_stocks_data(symbol)
        return JsonResponse({'status': 'success', 'message': f'Data for {symbol} fetched and saved.'})
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
def backtest_view(request):
    symbol = request.GET.get('symbol', 'AAPL')
    initial_investment = float(request.GET.get('initial_investment', 10000))

    try:
        result = backtest_strategy(symbol, initial_investment)
        return JsonResponse({'status': 'success', 'result': result})
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
def predict_stock_view(request):
    symbol = request.GET.get('symbol', 'AAPL')
    days = int(request.GET.get('days', 30)) 

    try:
        predictions = predict_stock_prices(symbol, days)
        return JsonResponse({'status': 'success', 'predictions': predictions})
    except ValueError as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

def report_view(request):
    report_type = request.GET.get('type', 'backtest')  # 'backtest' or 'prediction'
    symbol = request.GET.get('symbol', 'AAPL')

    try:
        # Backtest Report
        if report_type == 'backtest':
            initial_investment = float(request.GET.get('initial_investment', 10000))
            backtest_result = backtest_strategy(symbol, initial_investment)
            report = generate_backtest_report(backtest_result)

            # Generate the JSON response
            if request.GET.get('format') == 'json':
                return JsonResponse({'status': 'success', 'report': report})

            # Generate PDF using reportlab
            else:
                pdf_content = generate_pdf_report(report)
                response = HttpResponse(pdf_content, content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="backtest_report.pdf"'
                return response

        # Prediction Report
        elif report_type == 'prediction':
            days = int(request.GET.get('days', 30))
            predictions = predict_stock_prices(symbol, days)
            stock_data = pd.DataFrame(list(StockPrice.objects.filter(symbol=symbol).values('date', 'close_price')))
            future_dates = [prediction['date'] for prediction in predictions]
            predicted_prices = [prediction['predicted_price'] for prediction in predictions]

            # Generate report and plot
            report = generate_prediction_report(symbol, predictions)
            plot_image_base64 = plot_price_predictions(stock_data, predicted_prices, future_dates)

            # Include plot in the JSON report
            if request.GET.get('format') == 'json':
                report['price_chart'] = plot_image_base64
                return JsonResponse({'status': 'success', 'report': report})

            # Generate PDF with the chart image
            else:
                pdf_content = generate_pdf_report(report, plot_image_base64)
                response = HttpResponse(pdf_content, content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="prediction_report.pdf"'
                return response
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)