
import numpy as np
import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import datetime
from dateutil.relativedelta import relativedelta
from virgo_functions.configs import low_finder_configs

main_path = os.getcwd()


class slice_predict():

    def __init__(self, data, code, features, exeptions, scale_features, dates_back = 60 ):
    
        self.df = data
        self.code = code
        self.my_features = features
        self.exeptions = exeptions
        self.scale_features = scale_features
        self.dates_back = dates_back
        
        self.price = [ x for x in self.df.columns if '_price' in x ][0]
        self.target = [ x for x in self.df.columns if '_logdif' in x ][0]
        self.std_col = [ x for x in self.df.columns if '_stv' in x ][0]
        self.volume_col = [ x for x in self.df.columns if '_Volume' in x ][0]
        self.roll_mean_col = [ x for x in self.df.columns if '_roll_mean' in x ][0]

    def feature_engineering(self):
        df = (self.df
            .assign(up_yield = np.where(self.df[self.target] > 0, 1,0))
            .assign(low_yield = np.where(self.df[self.target] <= 0, 1,0))
        )
        
        df = df.rename(columns = {self.price:'price'})
        df["roll_up_yield"] = df.sort_values('Date')["up_yield"].transform(lambda x: x.rolling(10, min_periods=1).sum())
        df["roll_low_yield"] = df.sort_index()["low_yield"].transform(lambda x: x.rolling(10, min_periods=1).sum())
        df["roll_std"] = df.sort_index()[self.std_col].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['log_Volume'] = np.log(df[self.volume_col])
        df["roll_log_Volume"] = df.sort_index()['log_Volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        self.df = df
    def get_slice(self):
        
        begin_date = datetime.date.today()- relativedelta(days = self.dates_back)
        
        ds = self.df[self.df.Date >= begin_date]
        ds = ds.rename(columns = {self.price:'price'})

        ds_max = ds[ds[self.roll_mean_col] == ds[self.roll_mean_col].max()].head(1).Date.values[0]
        ds_min = ds[ds[self.roll_mean_col] == ds[self.roll_mean_col].min()].head(1).Date.values[0]
        ds['time_to_max'] = pd.to_numeric((self.df.Date - ds_max).dt.days,downcast='float')
        ds['time_to_min'] = pd.to_numeric((self.df.Date - ds_min).dt.days,downcast='float')
        
        ### apply pipeline sklearn

        X_train = ds[self.my_features + self.exeptions]

        pipeline = Pipeline([
            ('scaler', ColumnTransformer([('scaling', StandardScaler(), self.scale_features)], remainder='passthrough'))
        ])

        pipeline.fit(X_train)
        self.pipeline = pipeline
        self.ds = ds
        
        self.X_train_transformed = pipeline.transform(X_train)


features = low_finder_configs.features
exeptions = low_finder_configs.exeptions
scale_features = low_finder_configs.scale_features

def apply_feature_engine_bidfinder():

    raw_data = pd.read_csv(f'{main_path}/tmp_data/raw_data_bidfinder.csv')
    stock_codes = raw_data.stock_code.unique()

    dataframes = list()

    for code in stock_codes:
        stock_raw_data = raw_data[raw_data.stock_code == code]
        stock_raw_data = stock_raw_data.drop(columns = ['stock_code'])

        data_to_predict = slice_predict( stock_raw_data, code, features, exeptions, scale_features)
        data_to_predict.feature_engineering()
        data_to_predict.get_slice()
        dataset_to_predict = pd.DataFrame(data_to_predict.X_train_transformed, columns = features + exeptions)
        dataset_to_predict['stock_code'] = code
        
        dataframes.append(dataset_to_predict)
    
    target_data_export = pd.concat(dataframes)
    target_data_export.to_csv(f'{main_path}/tmp_data/dataset_to_predict_bidfinder.csv', header = True, index = False)
