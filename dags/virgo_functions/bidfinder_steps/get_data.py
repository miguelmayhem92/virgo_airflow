import yfinance as yf
import pandas as pd

import datetime
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from virgo_functions.configs import data_configs
import os
import numpy as np


def execute_raw_data_bidfinder():
    
    main_path = os.getcwd()
    stock_codes = data_configs.stock_codes
    n_days = data_configs.n_days
    window = data_configs.window
    ref_price = data_configs.ref_price
    logdif_column = data_configs.logdif_column
    std_column = data_configs.std_column
    lag_days = data_configs.lags

    datasets = list()

    def get_stock_data(stock_code, n_days, window, lags):
        today = datetime.date.today()
        begin_date = today - relativedelta(days = n_days)
        begin_date_str = begin_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(stock_code)
        df = stock.history(period="max")
        df = df.sort_values('Date')
        df.reset_index(inplace=True)
        
        ### getting rolling mean
        df["Close_roll_mean"] = (
            df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        
        ### getting rolling stdv
        df["Close_roll_std"] = (
            df.sort_values("Date")["Close"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        df["upper"] = df['Close_roll_mean'] + df["Close_roll_std"]*2
        df["lower"] = df['Close_roll_mean'] - df["Close_roll_std"]*2
        
        ### differencial analysis
        df['lag'] = df.Close.shift(lags)
        df['Dif'] = np.log(df['Close']) - np.log(df['lag'])
        df['Pos'] = np.where(df['Dif'] >= 0,df['Dif'], np.nan )
        df['Neg'] = np.where(df['Dif'] < 0,df['Dif'], np.nan )
        
        df = df[df.Date >= begin_date_str ]
        
        return df

    def shape_data(data, prefix, ref_price, std_column, logdif_column):
        data = data[['Date', ref_price, std_column, logdif_column, 'Volume','Close_roll_mean']]
        data = data.rename(columns = {
            ref_price: f'{prefix}_price',
            std_column: f'{prefix}_stv',
            logdif_column: f'{prefix}_logdif',
            'Volume':f'{prefix}_Volume',
            'Close_roll_mean':f'{prefix}_roll_mean'
        })
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d',utc=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        return data
    
    for code in stock_codes:
        data_stock = get_stock_data(stock_code = code, n_days = n_days, window = window, lags = lag_days)
        data_stock_ = shape_data(data_stock, 'stock', ref_price, std_column, logdif_column)
        data_stock_['stock_code'] = code
        datasets.append(data_stock_)
    
    raw_data = pd.concat(datasets)
    raw_data.to_csv(f'{main_path}/tmp_data/raw_data_bidfinder.csv', header = True, index = False)