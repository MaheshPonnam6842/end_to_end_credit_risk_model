import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join("artifacts","train.csv")
    test_data_path: str= os.path.join("artifacts","test.csv")
    raw_data_path: str= os.path.join("artifacts","data.csv")
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")

        try:
            BASE_DIR = os.getcwd()

            bureau_path = os.path.join(BASE_DIR, "notebook", "data", "bureau_data.csv")
            customers_path = os.path.join(BASE_DIR, "notebook", "data", "customers.csv")
            loans_path = os.path.join(BASE_DIR, "notebook", "data", "loans.csv")

            bureau = pd.read_csv(bureau_path)
            customers = pd.read_csv(customers_path)
            loans = pd.read_csv(loans_path)
            
            #merge data
            df = bureau.merge(customers, on="cust_id", how="inner")
            df = df.merge(loans, on="cust_id", how="inner")
            
            logging.info("Data Merged")
            
            logging.info("Minimal Data Cleaning")
            df = df.drop_duplicates()
            df["default"] = df["default"].astype(int)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)

            logging.info("Train Test Split Initiated")

            train_df, test_df= train_test_split(df, test_size= 0.25, random_state= 42, stratify=df["default"])

            train_df.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            test_df.to_csv(self.ingestion_config.test_data_path, index= False, header= True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__== "__main__":
    obj= DataIngestion()
    obj.initiate_data_ingestion()
        


                


