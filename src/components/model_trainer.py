import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Initiating Model Trainer")
            logging.info("Adding Smote to handle data imbalance")
            logging.info("Applying SMOTE on training data")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train, y_train
            )
            logging.info(f"Original class distribution: {np.bincount(y_train)}")

            logging.info(
                f"After SMOTE class distribution: {np.bincount(y_train_resampled)}"
            )
            # Defining models to train
            models = {
                "LogisticRegression": LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1
                ),
                "XGBoost": XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
                )
            }

            model_scores={}
            # Training and evaluating models
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train_resampled, y_train_resampled)

                logging.info(f"Evaluating {model_name}")
                auc= evaluate_model(X_test, y_test, model)
                model_scores[model_name] = auc
                
                logging.info(f"{model_name} AUC ROC Score: {auc}")
                

            # Selecting the best model
            best_model_name = max(model_scores,key=model_scores.get)
            best_model = models[best_model_name]
            best_model_score = model_scores[best_model_name] 

            if best_model_score <= 0.6:
                raise CustomException("No best model found with AUC ROC Score greater than the threshold", sys)  
            
            logging.info(f"Best Model: {best_model_name} with AUC ROC Score: {best_model_score}")

            # Saving the best model
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            logging.info("Model Trainer is completed")
            y_test_pred = best_model.predict(X_test)
            logging.info(f"Classification Report for best model:\n {classification_report(y_test, y_test_pred)}")
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)