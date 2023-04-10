from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def send_virgo_dashboard():
    gauth = GoogleAuth()           
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    key_id = '1OeHp-PDisvt3Et6CzlUK6J5o8d_Omm9R'

    root = os.getcwd()
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
