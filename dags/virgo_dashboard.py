from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator

from virgo_functions.get_data import execute_raw_data
from virgo_functions.feature_engine import apply_feature_engine
from virgo_functions.forecast_predict import apply_forecast

from datetime import datetime

with DAG(
    'virgo_dashboard', 
    start_date = datetime(2023,1,1),
    schedule_interval = "@daily",
    catchup = False
    ) as dag:

    start= DummyOperator(task_id='start')

    get_raw_data = PythonOperator(
        task_id='extract_raw_data',
        python_callable= execute_raw_data
        )

    get_feat_eng_data = PythonOperator(
        task_id='apply_feature_engineering',
        python_callable= apply_feature_engine
        )

    get_forecasting = PythonOperator(
        task_id='forecast',
        python_callable= apply_forecast
        )
    
    end= DummyOperator(task_id='end')

    start >> get_raw_data  >> get_feat_eng_data >> get_forecasting  >> end