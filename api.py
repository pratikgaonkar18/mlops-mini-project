from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Create FastAPI app
app = FastAPI(title="MLOps Mini Project API")

# Load production model from MLflow Model Registry
MODEL_URI = "models:/iris-classifier@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "MLOps Mini Project API is running"}

@app.post("/predict")
def predict(input: IrisInput):
    # Convert input to DataFrame
    data = pd.DataFrame([{
        "sepal length (cm)": input.sepal_length,
        "sepal width (cm)": input.sepal_width,
        "petal length (cm)": input.petal_length,
        "petal width (cm)": input.petal_width,
    }])

    # Make prediction
    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}
