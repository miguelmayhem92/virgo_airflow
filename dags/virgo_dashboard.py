from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator

from virgo_functions.forecasting_steps.get_data import execute_raw_data_forecast 
from virgo_functions.forecasting_steps.feature_engine import apply_feature_engine_forecast 
from virgo_functions.forecasting_steps.forecast_predict import apply_forecast 

from virgo_functions.bidfinder_steps.get_data import execute_raw_data_bidfinder
from virgo_functions.bidfinder_steps.feature_engine import apply_feature_engine_bidfinder
from virgo_functions.bidfinder_steps.forecast_predict import apply_bidfinder 

from virgo_functions.dashboard_steps.create_dashboard import execute_dashboard
from virgo_functions.dashboard_steps.send_dashboard import send_virgo_dashboard

from datetime import datetime

with DAG(
    'virgo_dashboard', 
    start_date = datetime(2023,1,1),
    schedule_interval = "@daily",
    catchup = False
    ) as dag:

    start= DummyOperator(task_id='start')

    ### forecasting flow
    
    get_raw_data_forecast = PythonOperator(
        task_id='extract_raw_data_forecast',
        python_callable= execute_raw_data_forecast
        )

    get_feat_eng_data_forecast = PythonOperator(
        task_id='apply_feature_engineering_forecast',
        python_callable= apply_feature_engine_forecast
        )

    get_forecasting = PythonOperator(
        task_id='forecast',
        python_callable= apply_forecast
        )

    ### bid finder flow

    get_raw_data_bidfinder = PythonOperator(
        task_id='extract_raw_data_bidfinder',
        python_callable= execute_raw_data_bidfinder
        )

    get_feat_eng_data_bidfinder = PythonOperator(
        task_id='apply_feature_engineering_bidfinder',
        python_callable= apply_feature_engine_bidfinder
        )

    get_bidfinder = PythonOperator(
        task_id='find_bid',
        python_callable= apply_bidfinder
        )

    #### dashboarding
    
    create_dashboard = PythonOperator(
        task_id="create_dashboard",
        python_callable= execute_dashboard
        )
    
    send_dashboard = PythonOperator(
        task_id='send_dashboard',
        python_callable= send_virgo_dashboard
    )

    #####

    end= DummyOperator(task_id='end')

    #####

    start >> get_raw_data_forecast  >> get_feat_eng_data_forecast >> get_forecasting
    start >> get_raw_data_bidfinder >> get_feat_eng_data_bidfinder >> get_bidfinder           

    [get_forecasting, get_bidfinder] >> create_dashboard >> send_dashboard >> end

    # send_dashboard >> end