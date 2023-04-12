from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import pandas as pd

root = os.getcwd()

def send_virgo_dashboard():

    root = os.getcwd()
    tmp = pd.read_csv(f'{root}/tmp_data/forecasts.csv')
    print(len(tmp))

    with open(f'{root}/tmp_data/client_secrets.json', 'r') as myfile:
        secretes = myfile.read()
        print('read json')

    gauth = GoogleAuth()       
    gauth.DEFAULT_SETTINGS['client_config_file'] = f'{root}/tmp_data/client_secrets.json'
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    key_id = '1OeHp-PDisvt3Et6CzlUK6J5o8d_Omm9R'
    #root = "C:/Users/Miguel/virgo_airflow/tmp_data/"
    filename = '/tmp_data/stocks_dashboard.html'

    upload_file= f'{root}{filename}'


    gfile = drive.CreateFile(
        {
            'title': filename,
            'parents': [{'id': key_id}]
        }
    )

    gfile.SetContentFile(upload_file)
    gfile.Upload()
