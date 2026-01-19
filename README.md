# Medical Insurance Cost Prediction API

## Project Overview

This project is an end-to-end Data Science application developed as part
of **CODTECH Data Science Internship -- Task 3**.

The goal of this project is to predict **medical insurance costs** based
on user information such as age, BMI, smoking habits, number of
children, and region. The trained Machine Learning model is deployed as
a **REST API using FastAPI**.

------------------------------------------------------------------------

## Problem Statement

Medical insurance costs vary based on multiple personal and lifestyle
factors. This project uses historical insurance data to build a
regression model that estimates insurance charges for new users.

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas, NumPy
-   Scikit-learn
-   FastAPI
-   Uvicorn
-   Joblib

------------------------------------------------------------------------

## Project Structure

    DS_Project/
    │
    ├── data/
    │   └── insurance.csv
    │
    ├── model/
    │   └── medical_cost_model.pkl
    │
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md

------------------------------------------------------------------------

## Dataset

-   **Source:** Kaggle -- Medical Cost Personal Dataset
-   **Features:**
    -   age
    -   sex
    -   bmi
    -   children
    -   smoker
    -   region
-   **Target Variable:** charges (medical insurance cost)

------------------------------------------------------------------------

## Machine Learning Model

-   Algorithm: **Linear Regression**
-   Data preprocessing done using **Scikit-learn Pipelines**
-   Categorical features encoded using **One-Hot Encoding**
-   Model saved using **Joblib** for deployment

------------------------------------------------------------------------

## How to Run the Project

### 1️. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2️. Train the model

``` bash
python train_model.py
```

### 3️. Start the API server

``` bash
python -m uvicorn app:app --reload
```

### 4️. Open in browser

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## API Endpoints

### 🔹 Home

    GET /

Response:

``` json
{
  "message": "Medical Insurance Cost Prediction API is running"
}
```

### 🔹 Predict Insurance Cost

    POST /predict
    POST /docs

#### Sample Input

``` json
{
  "age": 30,
  "sex": "male",
  "bmi": 27.5,
  "children": 1,
  "smoker": "no",
  "region": "southeast"
}
```

#### Sample Output

``` json
{
  "predicted_insurance_cost": 9124.67
}
```

------------------------------------------------------------------------

## Results

The model successfully predicts insurance costs based on user inputs and
provides real-time predictions through an API interface.

------------------------------------------------------------------------

## Conclusion

This project demonstrates a complete **end-to-end data science
workflow**, including: - Data preprocessing - Model training - Model
evaluation - API deployment

It fulfills all requirements of **CODTECH Internship Task 3**.

------------------------------------------------------------------------

## Author

**Pranav Mali**\
CODTECH Data Science Intern
