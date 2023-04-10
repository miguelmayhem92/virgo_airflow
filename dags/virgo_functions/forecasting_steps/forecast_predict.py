import pandas as pd
import os
from virgo_functions.prepro_data_forecast import split_data, WindowGenerator
from virgo_functions.configs import data_configs, low_finder_configs
from virgo_functions.mlflow_functions import call_params_from_mlflow

split_config = data_configs.split_config
features = low_finder_configs.features
main_path = os.getcwd()

class call_mlflow():
    def __init__(self, stock_code):
        self.stock_code = stock_code
        call_mlflow.call_params_from_mlflow = call_params_from_mlflow

def apply_forecast():

    target_data = pd.read_csv(f'{main_path}/tmp_data/target_data.csv')
    raw_data = pd.read_csv(f'{main_path}/tmp_data/raw_data.csv')
    stock_codes = target_data.stock_code.unique()

    dataframes = list()

    for stock_code in stock_codes:
        stock_target_data = target_data[target_data.stock_code == stock_code]
        stock_target_data = stock_target_data.drop(columns = ['stock_code'])

        stock_raw_data = raw_data[raw_data.stock_code == stock_code]
        stock_raw_data = stock_raw_data.drop(columns = ['stock_code'])

        split_object = split_data(stock_target_data, split_config)
        mlflow_object = call_mlflow(stock_code)
        mlflow_object.call_params_from_mlflow(call_model=True)

        column_indices = split_object.column_indices
        n = split_object.ndata
        train_df = split_object.train_df
        val_df  = split_object.val_df
        test_df = split_object.test_df
        num_features = split_object.num_features
        ### scaling
        split_object.scaling()
        train_mean = split_object.train_mean
        train_std = split_object.train_std
        train_df = split_object.train_df
        val_df  = split_object.val_df
        test_df = split_object.test_df

        wide_window = WindowGenerator(
            total_data = stock_target_data, 
            raw_stock = stock_raw_data,
            train_df=train_df, 
            val_df=val_df, 
            test_df=test_df,
            input_width= mlflow_object.input_length, 
            label_width= mlflow_object.OUT_STEPS, 
            shift= mlflow_object.OUT_STEPS,
            label_columns=['stock_logdif']
        )

        predictions, data_ = wide_window.expected_return(
                plot_col='stock_logdif',
                model = mlflow_object.model,
                train_mean = train_mean, 
                train_std = train_std,
                )

        final_result, some_history, prep_predictions = wide_window.get_futur_prices(
            predictions= predictions,
            steps_futur = mlflow_object.OUT_STEPS, 
            stock_code= stock_code,
            OUT_STEPS= mlflow_object.OUT_STEPS,
            train_std= train_std,
            train_mean = train_mean,
            lag_days= mlflow_object.lag_days,
            )
        final_result['stock_code'] = stock_code

        dataframes.append(final_result)
    
    dataset = pd.concat(dataframes)
    
    dataset.to_csv(f'{main_path}/tmp_data/forecasts.csv', header = True, index = False)