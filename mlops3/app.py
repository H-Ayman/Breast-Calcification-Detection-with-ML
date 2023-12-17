from fastapi import FastAPI, File, UploadFile, Response
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
custom_counter = Counter('my_custom_counter', 'Description of my custom counter')

# Define constants
num_features = 5  # Update with the actual number of features in your dataset

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
    return {"it's working properly, check prometheus for data visualisation"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
