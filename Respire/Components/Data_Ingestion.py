import os
import gdown
import zipfile
from Respire.Utils import *
from Respire.Logger import logging
from Respire.Exception import CustomException
from Respire.Entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self)-> str:
        try: 
            dataset_url, zip_download_dir = self.config.Source_URL, self.config.Local_Data_File
            os.makedirs("Artifacts/Data_Ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            try:
                prefix = 'https://drive.google.com/uc?export=download&id='
                gdown.download(prefix+file_id, str(zip_download_dir), quiet=False)
            except Exception as ge:
                import requests
                logging.info(f"gdown download failed ({ge}), attempting requests fallback...")
                URL = "https://docs.google.com/uc?export=download"
                session = requests.Session()
                response = session.get(URL, params={'id': file_id, 'confirm': 't'}, stream=True)
                with open(zip_download_dir, 'wb') as f:
                    for chunk in response.iter_content(32768):
                        if chunk:
                            f.write(chunk)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            import sys
            raise CustomException(e, sys)
        
    
    def extract_zip_file(self):
        try:
            unzip_path = self.config.Unzip_Dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.Local_Data_File, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logging.info(f"Extracted data from {self.config.Local_Data_File} into {unzip_path}")

        except Exception as e:
            raise CustomException(e)