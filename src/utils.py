import os
import pandas as pd
import sys
import numpy as np
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import roc_auc_score, classification_report

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_test, y_test, model):
    ''''
    This function using AUc ROC score to evaluate the model performance
    '''
    y_pred_proba= model.predict_proba(X_test)[:,1]
    auc= roc_auc_score(y_test, y_pred_proba)
    
    return auc
    