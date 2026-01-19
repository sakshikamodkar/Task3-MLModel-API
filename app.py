from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/medical_cost_model.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Predict medical insurance charges using ML model",
    version="1.0"
)

# Input schema
class InsuranceInput(BaseModel):
    age: int
    sex: str          # male / female
    bmi: float
    children: int
    smoker: str       # yes / no
    region: str       # southeast, southwest, northeast, northwest

# Home route
@app.get("/")
def home():
    return {"message": "Medical Insurance Cost Prediction API is running"}

# Prediction route
@app.post("/predict")
def predict_cost(data: InsuranceInput):

    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region": data.region
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    return {
        "predicted_insurance_cost": round(float(prediction), 2)
    }
