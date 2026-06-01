from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris classifier API")

model = joblib.load("models/model.pkl")

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array([[
        request.sepal_length, request.sepal_width,
        request.petal_length, request.petal_width
    ]])

    prediction = int(model.predict(features)[0])
    class_names = ["sentosa","versicolor","virginica"]
    probabilities = model.predict_proba(features)[0].tolist()


    return {
        "prediction": prediction,
        "class_name": class_names[prediction],
        "probabilities": probabilities
    }

@app.get("/health")
def health():
    return {"status": "healthy"}