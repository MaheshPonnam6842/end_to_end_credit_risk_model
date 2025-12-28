import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        # Must match training exactly
        self.selected_features = [
            "age",
            "income",
            "loan_amount",
            "loan_to_income",
            "loan_tenure_months",
            "avg_dpd_per_delinquency",
            "delinquency_ratio",
            "credit_utilization_ratio",
            "number_of_open_accounts",
            "residence_type",
            "loan_purpose",
            "loan_type"
        ]

    def _normalize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["residence_type"] = df["residence_type"].str.strip().str.lower()
        df["loan_purpose"] = df["loan_purpose"].str.strip().str.lower()
        df["loan_type"] = df["loan_type"].str.strip().str.lower()
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
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

        return df

    def predict(self, features: pd.DataFrame):
        try:
            features = features.copy()

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            features = self._normalize_categoricals(features)
            features = self._feature_engineering(features)

            features = features[self.selected_features]

            transformed_data = preprocessor.transform(features)

            prob_default = model.predict_proba(transformed_data)[:, 1][0]

            risk_label = "High Risk" if prob_default >= 0.5 else "Low Risk"

            return {
                "risk_label": risk_label,
                "default_probability": round(float(prob_default), 4),
                "default_probability_percent": round(float(prob_default) * 100, 2)
            }

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age: int,
                 income: float,
                 loan_amount: float,
                 loan_tenure_months: int,
                 
                 credit_utilization_ratio: float,
                 number_of_open_accounts: int,
                 total_loan_months: int,
                 delinquent_months: int,
                 total_dpd: float,
                 residence_type: str,
                 loan_purpose: str,
                 loan_type: str):
        
        self.age = age
        self.income = income
        self.loan_amount = loan_amount
        self.loan_tenure_months = loan_tenure_months
        
        self.credit_utilization_ratio = credit_utilization_ratio
        self.number_of_open_accounts = number_of_open_accounts
        self.total_loan_months = total_loan_months
        self.delinquent_months = delinquent_months
        self.total_dpd = total_dpd
        self.residence_type = residence_type
        self.loan_purpose = loan_purpose
        self.loan_type = loan_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "income": [self.income],
                "loan_amount": [self.loan_amount],
                "loan_tenure_months": [self.loan_tenure_months],
                
                "credit_utilization_ratio": [self.credit_utilization_ratio],
                "number_of_open_accounts": [self.number_of_open_accounts],
                "total_loan_months": [self.total_loan_months],
                "delinquent_months": [self.delinquent_months],
                "total_dpd": [self.total_dpd],
                "residence_type": [self.residence_type],
                "loan_purpose": [self.loan_purpose],
                "loan_type": [self.loan_type]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)