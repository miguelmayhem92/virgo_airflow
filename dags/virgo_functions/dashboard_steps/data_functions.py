import yfinance as yf
import pandas as pd
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta

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
    data = data[['Date', ref_price, std_column, logdif_column]]
    data = data.rename(columns = {
        ref_price: f'{prefix}_price',
        std_column: f'{prefix}_stv',
        logdif_column: f'{prefix}_logdif',
    })
    return data