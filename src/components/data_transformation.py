import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str= os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_obj(self,num_column,cat_column):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_pipeline= Pipeline(
                steps=[("imputer",SimpleImputer(strategy= "mean")),
                       ("standard_scaler", StandardScaler())
         ]
            )

            cat_pipeline= Pipeline([
                ("imputer", SimpleImputer(strategy= "most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ("standard_scaler", StandardScaler(with_mean=False))
            ])

            preprocessor= ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_column),
                    ("cat_pipeline", cat_pipeline, cat_column)
                ]
            )
            logging.info("Preprocessor is returned")
            return preprocessor
            

            
        except Exception as e:
            raise CustomException(e, sys)
        
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading Train and Test Data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test data completed")

            target_column= "default"
            logging.info("Adding features Feature Engineering")
            for df in [train_df, test_df]:

                df["loan_to_income"] = df["loan_amount"] / df["income"]

                df["delinquency_ratio"] = np.where(
                df["total_loan_months"] == 0,
                0,
                df["delinquent_months"] / df["total_loan_months"]
                )

                df["avg_dpd_per_delinquency"] = np.where(
                df["delinquent_months"] == 0,
                0,
                df["total_dpd"] / df["delinquent_months"]
                    )
                
            logging.info("Feature engineering completed")

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            num_column= X_train.select_dtypes(include=["int32","float32","int64", "float64"]).columns
            
            cat_column= X_train.select_dtypes(include=["object"]).columns
            logging.info(f"Numerical columns count: {len(num_column)}")
            logging.info(f"Categorical columns count: {len(cat_column)}")

            preprocessor_obj= self.get_data_transformer_obj(num_column,
            cat_column
                )
            
            logging.info("Fitting preprocessor on training data")

            X_train_transformed=preprocessor_obj.fit_transform(X_train)
            X_test_transformed= preprocessor_obj.transform(X_test)

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            logging.info("Preprocessor saved successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values
                    )
        except Exception as e:
            raise CustomException(e, sys) 
                


