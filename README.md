# 🚀 End-to-End Credit Risk Prediction System (Production-Grade)

This repository contains a **production-ready, end-to-end Machine Learning system** for **credit risk prediction**, built using **industry-standard ML engineering, MLOps, and DevOps practices**.

The project goes beyond model building and focuses on **automation, deployment, and real-world production readiness**, similar to how ML systems are built and deployed in financial services and fintech organizations.

---

## 🧠 Problem Statement

Credit risk assessment is a critical function in banking and financial services.  
The objective of this project is to **predict the probability of loan default** using customer, loan, and repayment behavior data, and expose the model through a **production inference service**.

Instead of stopping at notebooks, this project demonstrates **full ownership of the ML lifecycle** — from data ingestion to automated cloud deployment.

---

## 🏗️ High-Level Architecture

Data Sources
↓
Feature Engineering & Selection
↓
Preprocessing Pipeline
↓
Model Training & Evaluation
↓
Model + Preprocessor Artifacts
↓
Dockerized Flask Inference Service
↓
CI/CD via GitHub Actions
↓
Amazon ECR (Private Registry)
↓
EC2 Production Deployment

markdown
Copy code

---
flowchart LR
    User[User / Browser] -->|HTTP Request| FlaskApp[Flask Web App]

    FlaskApp --> PredictPipeline[Prediction Pipeline]

    PredictPipeline --> Preprocessor[Preprocessor.pkl]
    PredictPipeline --> Model[Trained ML Model.pkl]

    Preprocessor --> FeatureEngineering[Feature Engineering + Encoding + Scaling]
    FeatureEngineering --> Model

    Model -->|Probability Output| FlaskApp
    FlaskApp -->|Risk Score + Label| User

    subgraph Training Pipeline
        Data[Raw Data] --> Ingestion[Data Ingestion]
        Ingestion --> Transformation[Data Transformation]
        Transformation --> SMOTE[SMOTE Balancing]
        SMOTE --> Training[Model Training (XGBoost)]
        Training --> Model
        Transformation --> Preprocessor
    end

    subgraph CI/CD
        GitHub[GitHub Repo] --> Actions[GitHub Actions]
        Actions --> ECR[AWS ECR]
        ECR --> EC2[AWS EC2]
    end

    EC2 --> FlaskApp

## 🔑 Key Highlights

- End-to-end ML lifecycle (ingestion → training → inference)
- Feature engineering aligned between training and inference
- Class imbalance handled using **SMOTE**
- Multiple models evaluated (Logistic Regression, Random Forest, XGBoost)
- Best model selected using **ROC-AUC**
- Probability-based predictions (not just 0/1 output)
- Fully Dockerized application
- **CI/CD pipeline** using GitHub Actions
- **Private container registry (Amazon ECR)**
- **Automated deployment to EC2 using a self-hosted runner**

---
## 📐 System Architecture

This diagram shows the end-to-end architecture of the Credit Risk Prediction system, 
covering training, inference, Dockerization, and AWS deployment.

![Project Architecture](assets/Project_Architecture.png)

## 📊 Model Performance

Below is the comparison of model performance (AUC-ROC) across different algorithms 
used during experimentation.

![Best Model Scores](assets/Best_model_scores.png)


## 📊 Machine Learning Details

### Feature Engineering
Domain-driven features such as:
- Loan-to-Income Ratio
- Delinquency Ratio
- Average DPD per Delinquency
- Credit Utilization Metrics

### Feature Selection
Final features selected using:
- Domain knowledge
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature importance
- Multicollinearity checks (VIF)

### Models Trained
- Logistic Regression (baseline, interpretable)
- Random Forest
- **XGBoost (final production model)**

### Evaluation Metrics
- ROC-AUC (primary metric)
- Precision, Recall, F1-score
- SMOTE applied to handle class imbalance

---

## 🧪 Training Pipeline

- Modular pipeline structure
- Saved artifacts:
  - `model.pkl`
  - `preprocessor.pkl`
- Strict consistency between training and inference pipelines
- Reproducible training workflow

---

## 🌐 Inference Application

- Flask-based web application
- User-friendly form input
- Outputs:
  - Risk classification (High / Low)
  - Probability of default (percentage)
- Clean separation between:
  - API logic
  - Prediction logic
  - Feature processing

---

## 🐳 Dockerization

- Application fully containerized using Docker
- Same image used across:
  - Local testing
  - CI/CD pipeline
  - Production deployment
- No dependency on local environment

---

## 🔄 CI/CD Pipeline (GitHub Actions)

On every push to the `main` branch:

1. **Continuous Integration**
   - Code checkout
   - Lint / test stage (extendable)

2. **Continuous Delivery**
   - Docker image build
   - Image pushed to **Amazon ECR**

3. **Continuous Deployment**
   - Self-hosted GitHub runner on EC2
   - Pulls latest image from ECR
   - Stops old container
   - Runs new container automatically

**Deployment is fully automated with zero manual intervention.**

---

## ☁️ Cloud & Infrastructure

- **Amazon EC2** – Compute
- **Amazon ECR** – Private container registry
- **IAM User** – Secure access (upgradeable to IAM Role)
- **Security Groups** – Controlled ingress (SSH + App port)

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

### Backend & Deployment
- Flask
- Docker
- GitHub Actions
- Amazon EC2
- Amazon ECR
- Linux (Ubuntu / Amazon Linux)

---

## 📁 Project Structure

## 📁 Project Structure

```text
.
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
└── README.md
```

### Structure Overview
- `src/components` – Core ML components (ingestion, transformation, training)
- `src/pipeline` – Training and inference pipelines
- `artifacts` – Serialized model and preprocessing objects
- `templates` – Flask UI templates
- `Dockerfile` – Containerization configuration
- `.github/workflows` – CI/CD pipeline using GitHub Actions



yaml
Copy code

---

## 🚀 Run Locally Using Docker

```bash
docker build -t credit-risk-app .
docker run -p 5000:5000 credit-risk-app
Access the application at:

arduino
Copy code
http://localhost:5000
📌 Production Deployment
Docker image stored in Amazon ECR

Automatically deployed on EC2

Triggered via GitHub Actions on every push to main

🔮 Future Enhancements
SHAP-based explainability for single predictions

Gunicorn + Nginx for high-throughput serving

ECS / Auto-scaling deployment

Monitoring with CloudWatch

Model versioning and rollback strategy

👤 Author
Mahesh Ponnam
Data Scientist | Machine Learning | MLOps

Focused on building production-ready ML systems, not just notebooks.
