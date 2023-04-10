import pandas as pd
import os
from virgo_functions.configs import data_configs, low_finder_configs
from virgo_functions.mlflow_functions import  call_bidfinder_model

split_config = data_configs.split_config
features = low_finder_configs.features
main_path = os.getcwd()

def apply_bidfinder():

    class predict():
            def __init__(self, model, data,X_test):
                
                self.model = model
                self.data = data
                self.X_test = X_test
                self.y_pred = self.model.predict(self.X_test)
                self.data['prediction'] = self.y_pred

    model = call_bidfinder_model()

    dataset_to_predict_bidfinder = pd.read_csv(f'{main_path}/tmp_data/dataset_to_predict_bidfinder.csv')
    stock_codes = dataset_to_predict_bidfinder.stock_code.unique()

    results = list()

    for code in stock_codes:
        dataset_to_predict = dataset_to_predict_bidfinder[dataset_to_predict_bidfinder.stock_code == code]
        dataset_to_predict = dataset_to_predict.drop(columns = ['stock_code'])

        X_predict = dataset_to_predict[features]
        predictions = predict(model, dataset_to_predict, X_predict)

        result = predictions.data
        result['stock'] = code
        result = result[result.prediction == 1][['Date','prediction','stock']]

        results.append(result)
    
    results = pd.concat(results)
    results.to_csv(f'{main_path}/tmp_data/bids-predictions.csv', header = True, index = False)
