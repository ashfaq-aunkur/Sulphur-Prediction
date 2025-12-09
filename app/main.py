from fastapi import FastAPI            
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import json

models, params = {}, {}

def apply_scaler_X(df_num, scaler_json):
    mean = np.array(scaler_json["mean"])
    scale = np.array(scaler_json["scale"])
    return (df_num - mean) / scale

def apply_encoder_X(df_cat, encoder_json):
    categories = encoder_json["categories"]
    encoded_arrays = []
    for i, col in enumerate(df_cat.columns):
        categories_i = categories[i]
        value = df_cat.iloc[0, i]
        one_hot = [1 if value == cat else 0 for cat in categories_i]
        encoded_arrays.append(one_hot)
    return np.array(encoded_arrays).flatten().reshape(1, -1)

async def load_model():
    models["S_model"] = tf.keras.models.load_model("app/models/s_prediction_model.keras")

async def load_params():
    with open("app/params/scaler_X.json", "r") as f:
        params["scaler_X"] = json.load(f)
    
    with open("app/params/encoder_X.json", "r") as f:
        params["encoder_X"] = json.load(f)

    with open("app/params/feature_info.json", "r") as f:
        params["feature_info"] = json.load(f)

class TestInput(BaseModel):
    AEZ: int
    N: float
    P: float
    K: float
    pH: float
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    await load_params()
    yield
    models.clear()    
    params.clear()

app = FastAPI(title="Sulphur Prediction",lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_S")
async def predict_S(input_data: TestInput):
    model = models["S_model"]
    encoder_X = params["encoder_X"]  
    scaler_X = params["scaler_X"]
    feature_info = params["feature_info"]

    df = pd.DataFrame([{
        "n": input_data.N,
        "p": input_data.P,
        "k": input_data.K,
        "ph": input_data.pH,
        "aez": input_data.AEZ
    }])

    numeric_cols = feature_info["numeric_features"]
    categorical_cols = feature_info["categorical_features"]

    X_num = apply_scaler_X(df[numeric_cols], scaler_X)
    X_cat = apply_encoder_X(df[categorical_cols], encoder_X)
    X_processed = np.hstack([X_num, X_cat]).reshape(1, -1)

    prediction = model.predict(X_processed)

    return {"predicted_S": float(prediction[0][0])}