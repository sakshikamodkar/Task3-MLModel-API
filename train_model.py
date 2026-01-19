import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("data/insurance.csv")

# Separate features and target
X = data.drop("charges", axis=1)
y = data["charges"]

# Categorical and numerical columns
categorical_cols = ["sex", "smoker", "region"]
numerical_cols = ["age", "bmi", "children"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = LinearRegression()

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully")
print(f"📉 Mean Absolute Error: {mae:.2f}")
print(f"📈 R2 Score: {r2:.2f}")

# Save model
joblib.dump(pipeline, "model/medical_cost_model.pkl")
print("💾 Model saved as medical_cost_model.pkl")
