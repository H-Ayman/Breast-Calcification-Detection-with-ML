from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import joblib
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# Load the saved model
model = joblib.load('knn_model.joblib')

Instrumentator().instrument(app).expose(app)

# Create a counter metric to track HTTP requests
http_requests_total = Counter('http_requests_total', 'Total number of HTTP requests')

# Define constants
num_features = 5  

class Item(BaseModel):
    features: list

@app.post("/predict")
async def predict(item: Item):
    features = np.array(item.features).reshape(1, -1)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    # Increment the counter on each request
    http_requests_total.inc()
    return {"it's": "working"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    # Return Prometheus-compatible metrics
    return generate_latest(http_requests_total)
