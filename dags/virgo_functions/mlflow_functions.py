import mlflow.pyfunc
from mlflow import MlflowClient
import os
#import tensorflow


def call_params_from_mlflow(self, call_model = False):

        registered_model_name = f'{self.stock_code}_models'
        filter_registered_model_name = "name='{}'".format(registered_model_name)

        client = MlflowClient()
        latest_version_info = client.get_latest_versions(registered_model_name, stages=["Production"])
        latest_production_version = latest_version_info[0].version

        for mv in client.search_model_versions(filter_string = f'{filter_registered_model_name}'):
            mv = dict(mv)
            if mv['version'] == latest_production_version:
                run_id = mv['run_id']
                break

        model_params = client.get_run(run_id).data.params
        self.lag_days = int(model_params['lag_days'])
        self.input_length = int(model_params['input_length'])
        self.OUT_STEPS = int(model_params['OUT_STEPS'])

        model_version = latest_production_version
        self.model_version = model_version
        print(self.model_version, self.lag_days, self.input_length, run_id)

        if call_model:
            
            path = os.getcwd()
            model = mlflow.pyfunc.load_model(
                model_uri=f"{path}/mlruns/{run_id}/artifacts/{self.stock_code}-run",
                suppress_warnings = True
            )
            print('method 3 worded')

            self.model = model 