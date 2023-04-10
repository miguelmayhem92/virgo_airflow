import yfinance as yf
import pandas as pd

import datetime
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from virgo_functions.configs import data_configs
import os


def execute_raw_data_forecast():

    main_path = os.getcwd()

    stock_codes = data_configs.stock_codes
    n_days = data_configs.n_days
    window = data_configs.window

    class get_raw_data():
        
        def __init__(self, stock_code):
            self.stock_code = stock_code

        def get_basic_feaures(self,prefix, n_days, window):
            
            self.prefix = prefix
            
            today = datetime.date.today()
            begin_date = today - relativedelta(days = n_days)
            begin_date_str = begin_date.strftime('%Y-%m-%d')
            
            stock = yf.Ticker(self.stock_code)
            df = stock.history(period="max")
            df = df.sort_values('Date')
            df.reset_index(inplace=True)
            print(len(df))
            
            ### getting rolling stdv
            df["Close_roll_std"] = (
                df.sort_values("Date")["Close"]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            
            ### applying smoothing
            smoother = ExponentialSmoothing(df.Close)
            df['smoothed_Close'] = smoother.fit(smoothing_level=0.3,smoothing_trend=0.1,smoothing_seasonal=0, optimized=False).fittedvalues
            
            df = df[df.Date >= begin_date_str ]
            
            data = df[['Date','Close', 'Close_roll_std', 'Volume', 'smoothed_Close']]
            data = data.rename(columns = {
                'Close': f'{self.prefix}_price',
                "Close_roll_std": f'{self.prefix}_stv',
                'Volume':f'{self.prefix}_Volume',
                'Close_roll_mean':f'{self.prefix}_roll_mean'
            })
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d',utc=True)
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            data['stock_code'] = self.stock_code
            self.raw_data = data

    datasets = list()

    for stock_code in stock_codes:

        stock_data_object = get_raw_data(stock_code)
        stock_data_object.get_basic_feaures(prefix = 'stock', n_days = n_days, window = window)
        raw_data = stock_data_object.raw_data
        datasets.append(raw_data)

    dataset = pd.concat(datasets)
    dataset.to_csv(f'{main_path}/tmp_data/raw_data.csv', header = True, index = False)

