from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import time
import logging
import joblib


# Create FastAPI app
app = FastAPI(title="MLOps Mini Project API")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("mlops-api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled error while processing request")
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"path={request.url.path} status={status_code if 'status_code' in locals() else 'ERROR'} "
            f"latency_ms={latency_ms:.2f}"
        )

# Load production model from MLflow Model Registry
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)


# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "MLOps Mini Project API is running ci-test"}

@app.get("/health")
def health():
    return {"status": "ok"}

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
