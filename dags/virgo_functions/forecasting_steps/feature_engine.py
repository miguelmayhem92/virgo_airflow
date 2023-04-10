
import numpy as np
import os
import pandas as pd
from virgo_functions.mlflow_functions import call_params_from_mlflow

main_path = os.getcwd()

class object_feature_eng():
    def __init__(self, stock_code, prefix,raw_data):
        self.stock_code = stock_code
        self.raw_data = raw_data
        self.prefix = prefix
        object_feature_eng.call_params_from_mlflow = call_params_from_mlflow

    def data_features_eng(self):
        
        std_col = [ x for x in self.raw_data.columns if '_stv' in x ][0]
        volume_col = [ x for x in self.raw_data.columns if '_Volume' in x ][0]
        
        self.raw_data["roll_std"] = self.raw_data.sort_index()[std_col].transform(lambda x: x.rolling(10, min_periods=1).mean())
        self.raw_data['log_Volume'] = np.log(self.raw_data[volume_col])
        self.raw_data["roll_log_Volume"] = self.raw_data.sort_index()['log_Volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        self.raw_data['noise_price'] = self.raw_data.smoothed_Close - self.raw_data.stock_price
        
        self.raw_data = (
                self.raw_data
                .assign(count_pos_noise = np.where(self.raw_data.noise_price > 0, 1,0))
                .assign(count_neg_noise = np.where(self.raw_data.noise_price <= 0, 1,0))
            )
        self.raw_data["roll_pos_noise_count"] = self.raw_data.sort_values('Date')["count_pos_noise"].transform(lambda x: x.rolling(10, min_periods=1).sum())
        self.raw_data["roll_neg_noise_count"] = self.raw_data.sort_values('Date')["count_neg_noise"].transform(lambda x: x.rolling(10, min_periods=1).sum())
        self.raw_data = self.raw_data.drop(columns = ['count_pos_noise', 'count_neg_noise'])

        def measure_distance(data, roll_scope, typex):

            if typex =='max':
                data["dist"] = data.sort_values('Date')['smoothed_Close'].transform(lambda x: x.rolling(roll_scope, min_periods=1).max())
            elif typex == 'min':
                data["dist"] = data.sort_values('Date')['smoothed_Close'].transform(lambda x: x.rolling(roll_scope, min_periods=1).min())

            data["dist_tmp"] = 0

            for i in range(len(data)):
                dist_ref = data.iloc[i,data.columns.get_loc("dist")]
                for j in range(0 if i-roll_scope <= 0 else i-roll_scope, i+1 ):

                    ref_value = data.iloc[j,data.columns.get_loc("smoothed_Close")]
                    if ref_value == dist_ref:
                        date_ref = data.iloc[j,data.columns.get_loc("Date")]
                        data.iloc[i,data.columns.get_loc("dist_tmp")] = date_ref
                        continue

            data[f'dist_{typex}'] =  pd.to_numeric((pd.to_datetime(data['Date']) - pd.to_datetime(data['dist_tmp'])).dt.days,downcast='float')
            data = data.drop(columns = ['dist','dist_tmp'])

            return data
        
        self.raw_data = measure_distance(data = self.raw_data, roll_scope = 10, typex = 'max')
        self.raw_data = measure_distance(data = self.raw_data, roll_scope = 10, typex = 'min')

    def data_target_features_eng(self, lags, list_lags = [30]):
        
        self.target_data = self.raw_data
        
        price = [ x for x in self.target_data.columns if '_price' in x ][0]
        
        self.target_data['lag'] = self.target_data.stock_price.shift(lags)
        self.target_data[f'{self.prefix}_logdif'] = np.log(self.target_data[price]) - np.log(self.target_data['lag'])
        
        self.target_data = self.target_data.rename(columns = {'Close': f'{self.prefix}_price'})
        target = f'{self.prefix}_logdif'
        
        self.target_data = (
            self.target_data
            .sort_values('Date')
            .set_index('Date')
            .assign(up_yield = np.where(self.target_data[target] > 0, 1,0))
            .assign(low_yield = np.where(self.target_data[target] <= 0, 1,0))
        )
        ## rolling operations
        self.target_data["roll_up_yield"] = self.target_data.sort_index()["up_yield"].transform(lambda x: x.rolling(3, min_periods=1).sum())
        self.target_data["roll_low_yield"] = self.target_data.sort_index()["low_yield"].transform(lambda x: x.rolling(3, min_periods=1).sum())
        self.target_data[f"roll_{target}"] = self.target_data.sort_index()[target].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        ## getting lags
        if list_lags:
            lags = list_lags
            columns_to_lag = [target,f"roll_{target}","roll_up_yield", "roll_low_yield"]

            for lag_ in lags:
                for col_ in columns_to_lag:
                    self.target_data[f'lag_{lag_}_{col_}'] = self.target_data[col_].shift(lag_)
        
        ## some cleaning
        self.target_data = (
            self.target_data
            .drop(columns = [price,'stock_stv','stock_Volume','lag','up_yield', 'low_yield','log_Volume','noise_price'])
            .dropna(axis='rows')
            .sort_index()
        )


def apply_feature_engine_forecast():

    raw_data = pd.read_csv(f'{main_path}/tmp_data/raw_data.csv')
    stock_codes = raw_data.stock_code.unique()

    dataframes = list()

    for stock_code in stock_codes:
        stock_raw_data = raw_data[raw_data.stock_code == stock_code]
        stock_raw_data = stock_raw_data.drop(columns = ['stock_code'])
        feat_engine_stock = object_feature_eng(stock_code = stock_code,prefix = 'stock', raw_data = stock_raw_data)
        feat_engine_stock.call_params_from_mlflow()
        feat_engine_stock.data_features_eng()
        feat_engine_stock.data_target_features_eng(lags = feat_engine_stock.lag_days)
        target_data = feat_engine_stock.target_data
        target_data['stock_code'] = stock_code
        dataframes.append(target_data)

    target_data_export = pd.concat(dataframes)
    target_data_export.to_csv(f'{main_path}/tmp_data/target_data.csv', header = True, index = False)