from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
application = Flask(__name__)
app = application

## Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data=CustomData(
            age = int(request.form.get('age')),
            income = float(request.form.get('income')) , 
            loan_amount = float(request.form.get('loan_amount')),
            loan_tenure_months = int(request.form.get('loan_tenure_months')),
            credit_utilization_ratio = float(request.form.get('credit_utilization_ratio')),
            number_of_open_accounts = int(request.form.get('number_of_open_accounts')),
            total_loan_months = int(request.form.get('total_loan_months')),
            delinquent_months = int(request.form.get('delinquent_months')),
            total_dpd = float(request.form.get('total_dpd')),
            residence_type = request.form.get('residence_type'),
            loan_purpose = request.form.get('loan_purpose'),
            loan_type = request.form.get('loan_type'),
            )
            pred_df= data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline= PredictPipeline()
            results= predict_pipeline.predict(pred_df)

            return render_template(
                "home.html",
                risk=results["risk_label"],
                probability=results["default_probability_percent"]
            )
        except Exception as e:
            return render_template('home.html', error_message=str(e))

    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000, debug=True)
    
    


            
