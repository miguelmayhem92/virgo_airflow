import yfinance as yf
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import datetime
from dateutil.relativedelta import relativedelta

from virgo_functions.dashboard_steps import data_functions, configs

import warnings
warnings.filterwarnings('ignore')

root = os.getcwd()
#root = "C:/Users/Miguel/virgo_airflow/tmp_data/"

delta_back = configs.optimazation_configs.days_back_predictions

n_days = configs.data_configs.n_days
lag_days = configs.data_configs.lags
window = configs.data_configs.window

today = datetime.date.today()
dates_str_vector = [(today - relativedelta(days = x)).strftime("%Y-%m-%d") for x in range(delta_back)]

def execute_dashboard():

    batch_predictions_csv = pd.read_csv(f'{root}/tmp_data/forecasts.csv')
    bid_finder_output = pd.read_csv(f'{root}/tmp_data/bids-predictions.csv') 

    stocks_codes_ = batch_predictions_csv.StockCode.unique() ## all

    data_history_stocks = dict()
    for stock_code_name in stocks_codes_:
        data_history_stocks[stock_code_name] = data_functions.get_stock_data(stock_code = stock_code_name, n_days = n_days, window = window, lags = lag_days)

    n_stocks = len(stocks_codes_ )
    subtitles = [ code + 'plot' for code in stocks_codes_]

    data_sets=data_history_stocks
    predictions = batch_predictions_csv
    bid_finder = bid_finder_output


    fig = make_subplots(rows=n_stocks, cols=1,vertical_spacing = 0.1,shared_xaxes=True,
                           subplot_titles=subtitles)

    for i,stock_code in enumerate(stocks_codes_):
        i = i+1
        df = data_sets[stock_code]
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d',utc=True).dt.date
        stock_prediction = predictions[(predictions['Type'] == 'Forecast') & (predictions['StockCode'] == stock_code)]
        
        bid_prediction = bid_finder[bid_finder.stock == stock_code]
        bid_prediction['Date'] = pd.to_datetime(bid_finder_output.Date).dt.date

        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close, marker_color = 'blue', name='Price'),row=i, col = 1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close_roll_mean, marker_color = 'grey', name='roll mean' ),row=i, col = 1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.lower, marker_color = 'pink', name='bound', legendgroup = '1' ),row=i, col = 1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df.upper, marker_color = 'pink', name='bound', showlegend=False, legendgroup = '1' ),row=i, col = 1)
        
        df = bid_prediction.merge(df[['Date','Close']], on = 'Date', how = 'left')
        df['Close'] = df['Close']*1.005
        fig.add_trace(go.Scatter(x=df['Date'], y=df.Close, name='Bid',mode='markers', line = dict(color = 'red')),row=i, col = 1)
        
        last_exe_prediction_date = stock_prediction.ExecutionDate.unique()
        last_date = max(last_exe_prediction_date)
        
        for i,datex in enumerate([x for x in last_exe_prediction_date if x != last_date]):
            df = stock_prediction[stock_prediction.ExecutionDate == datex]
            legend = True if i == 0 else False
            fig.add_trace(go.Scatter(x=df['Date'], y=df.stock_price, mode='markers', marker_color = 'green', name='past prediction', legendgroup = '0', showlegend = legend),row=i, col = 1)
            
        df = stock_prediction[stock_prediction.ExecutionDate == last_date]
        fig.add_trace(go.Scatter(x=df['Date'], y=df.stock_price, name='last prediction', line = dict(color = 'blue', dash='dash')),row=i, col = 1)

    fig.update_layout(height=500*n_stocks +100, width=900, title_text=f"stocks vizualization")
    # fig.show()

    fig.write_html(f"{root}/tmp_data/stocks_dashboard.html")