'''
# Reto de Data Science en DEACERO - Predicción de precio de varilla corrugada
# Script: Funciones para API HTTP RESTful

Tags: #API #FastAPI 

Autor: Alessio Daniel Hernández Rojas

Última actualización: 29-12-2025 (Creado: 28-12-2025)

Descripción: El presente script tiene como objetivo definir el flujo y métodos de la API HTTP RESTful para predecir el precio de varilla corrugada.

Entradas:
    * df_train: DataFrame preprocesado para entrenamiento  
    * df_live: DataFrame preprocesado para predicción 
    * modelo: Modelo entrenado

Salida: 
    * API HTTP RESTful con endpoint para predecir el precio de varilla corrugada

'''

from fastapi import FastAPI, Header, HTTPException, Depends
from collections import defaultdict
from fastapi.staticfiles import StaticFiles
from . import model
import random
from fastapi.responses import JSONResponse
import os
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(CURRENT_DIR, "..", "output", "app", "static")

app = FastAPI()

# Cargar y preparar al iniciar
df_live, df_train = model.pipeline_preprocesamiento()
modelo = model.preparar_modelo(df_train)
print("Vector a predecir:")
print(df_live.drop(columns=["target_t_plus_1"]).tail(1).T)





# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "service": "Steel Rebar Price Predictor",
        "version": "1.0",
        "documentation_url": "https://github.com/adhro/Reto_DEACERO",
        "data_sources": [
            "Yahoo Finance",
            "World Bank Commodity Prices",
            "Trading Economics"
        ],
        "last_model_update": datetime.utcnow().isoformat() + "Z"
    }

# =========================
# API KEY
# =========================
API_KEY = "deacero-2025"

def get_api_key(x_api_key: str = Header(...)):
    return x_api_key

def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# =========================
# RATE LIMIT
# =========================
rate_limit_store = defaultdict(list)

def rate_limiter(api_key: str):
    now = datetime.utcnow()
    window = timedelta(hours=1)

    rate_limit_store[api_key] = [
        t for t in rate_limit_store[api_key]
        if now - t < window
    ]

    if len(rate_limit_store[api_key]) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    rate_limit_store[api_key].append(now)

# =========================
# CACHE
# =========================
prediction_cache = {
    "timestamp": None,
    "value": None
}

# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.get("/predict/steel-rebar-price")
def predict_price(api_key: str = Depends(get_api_key)):

    verify_api_key(api_key)
    rate_limiter(api_key)

    now = datetime.utcnow()

    if (
        prediction_cache["timestamp"]
        and now - prediction_cache["timestamp"] < timedelta(hours=1)
    ):
        prediction = prediction_cache["value"]
    else:
        X_pred = df_live.drop(columns=["target_t_plus_1"]).tail(1)
        prediction = float(modelo.predict(X_pred)[0])
        prediction_cache["timestamp"] = now
        prediction_cache["value"] = prediction

    return JSONResponse(
        content={
            "prediction_date": (now + timedelta(days=1)).date().isoformat(),
            "predicted_price_usd_per_ton": round(prediction, 2),
            "currency": "USD",
            "unit": "metric ton",
            "model_confidence": 0.80,
            "timestamp": now.isoformat() + "Z"
        }
    )