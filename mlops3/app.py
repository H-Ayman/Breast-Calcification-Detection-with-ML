from fastapi import FastAPI, File, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import joblib
import numpy as np
import io
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the saved model
model = joblib.load('knn_model.joblib')

Instrumentator().instrument(app).expose(app)

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
