#####################################################################################################################################
'''
# Reto de Data Science en DEACERO - Predicción de precio de varilla corrugada
# Script: Calidad de datos, y Construcción de Modelo.

Tags: #Model # API #FastAPI #RESTful #DataCleaning #Preprocessing #PredictionModel #SteelRebarPricePrediction

Autor: Alessio Daniel Hernández Rojas

Última actualización: 29-12-2025 (Creado: 28-12-2025)

Descripción: El presente script tiene como objetivo realizar las funciones para un modelo desplegado por API HTTP RESTful para predicción de precio de varilla corrugada.

Entrada: 
    * Base de datos compuesta de distintas fuentes

Salidas: 
    * df_train: DataFrame preprocesado para entrenamiento  
    * df_live: DataFrame preprocesado para predicción 
    * modelo: Modelo entrenado

'''
#####################################################################################################################################
# model.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from sklearn.ensemble import IsolationForest
from math import radians, sin, cos, sqrt, atan2
from typing import Sequence
from sklearn.preprocessing import OneHotEncoder
import yfinance as yf
import datetime as dt
from pandas_datareader import data as pdr
from xgboost import XGBRegressor

# Imputación de missings
def imputar_missings(df: pd.DataFrame,
                                    ) -> pd.DataFrame:
    """
    Imputa features y separa datos de entrenamiento y producción
    """
    print(f"✅ Imputación de valores nulos. {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Imputa features
    x_cols = [c for c in df.columns if c.startswith(("num_", "tgt_rebar_price_lag"))]
    df[x_cols] = df[x_cols].ffill().bfill()
    
    # Elimina filas donde el target no existe
    df_train = df.dropna(subset=["target_t_plus_1"]).copy()

    # Datos para producción (NO eliminar filas)
    df_live = df.copy()

    print(
        f"📊 Train hasta {df_train.index.max().date()} "
        f"({len(df_train)} obs)"
    )
    print(
        f"🚀 Live hasta {df_live.index.max().date()} "
        f"({len(df_live)} obs)"
    )
        
    return df_live, df_train

# Feature engineering
def feature_engineering(df):

    # Guardar columnas originales
    original_cols = set(df.columns)

    lag_features = [
        "num_aluminum_price",
        "num_crude_oil_price",
        "num_copper_price",
        "num_hot_rolled_coil_price",
        "num_coking_coal_price",
        "num_iron_ore_price"
    ]

    for col in lag_features:
        for lag in [1, 3, 5, 7]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Target futuro
    df["target_t_plus_1"] = df["tgt_rebar_price"].shift(-1)

    # Eliminar columnas originales usadas
    df = df.drop(columns=lag_features + ["tgt_rebar_price"])

    # Detectar columnas nuevas
    new_cols = sorted(set(df.columns) - original_cols)
    
    print(f"✅ Feature engineering aplicada. {df.shape[0]} filas y {df.shape[1]} columnas.")
    print(f"📈 Nuevas columnas ({len(new_cols)}):")
    for c in new_cols:
        print(f"   - {c}")

    return df

# Filtro de variables unarias ponderadas
def filtro_unarias(df, umbral=0.95, mostrar_sesgo=True):
    """
    Detecta variables en las que una categoría supera el umbral de proporción
    y las elimina del DataFrame sobrescribiéndolo.
    
    Parámetros:
    - df: DataFrame original
    - umbral: proporción máxima permitida de una sola categoría
    - mostrar_sesgo: si True, imprime la distribución de las variables unarias
    
    Retorna:
    - df limpio sin las columnas unarias
    """
    unarias = []

    for v in df.columns:
        # Distribución de frecuencias relativas
        evaluacion = df[v].value_counts(normalize=True).reset_index()
        evaluacion.columns = [v, "proporcion"]

        # Verificar si la categoría dominante supera el umbral
        if evaluacion["proporcion"].iloc[0] > umbral:
            unarias.append(v)

            if mostrar_sesgo:
                print(f"\n🔍 Variable: {v}")
                print(evaluacion.to_string(index=False, formatters={"proporcion": "{:.2%}".format}))

    print(f'✅ Filtro de variables unarias aplicado con umbral {umbral}. {len(unarias)} columnas eliminadas.')
    if not unarias:
        print("✅ No se encontraron variables unarias por encima del umbral.")
    else:
        # Eliminar columnas unarias sobrescribiendo df
        exception = 'cat_property_type'
        cols_a_drop = [col for col in unarias if col != exception]
        df = df.drop(columns=cols_a_drop)
        print(f"📉 Columnas eliminadas por ser unarias: {unarias}")

    return df

# Filtro de completitud
def filtrar_por_completitud(df, umbral=90):
    """
    Filtra las columnas de un DataFrame que tengan una completitud menor al umbral indicado.
    Imprime las columnas eliminadas junto con su porcentaje de completitud.
    Devuelve el DataFrame filtrado.
    """
    # Calcular completitud
    comple = pd.DataFrame(df.isnull().sum())
    comple.reset_index(inplace=True)
    comple = comple.rename(columns={"index": "columna", 0: "total"})
    comple["completitud"] = (1 - comple["total"] / df.shape[0]) * 100

    # Identificar columnas que no cumplen el umbral
    eliminadas = comple.loc[comple["completitud"] < umbral, ["columna", "completitud"]]
    conservadas = comple.loc[comple["completitud"] >= umbral, "columna"]

    print(f'✅ Filtro de completitud aplicado con umbral {umbral}%. {len(eliminadas)} columnas eliminadas.')
    # Imprimir reporte
    if not eliminadas.empty:
        print("📉 Columnas eliminadas por baja completitud:")
        for _, row in eliminadas.iterrows():
            print(f" - {row['columna']}: {row['completitud']:.2f}% de completitud")
    else:
        print("✅ No se eliminaron columnas, todas cumplen el umbral.")

    # Retornar DataFrame filtrado
    return df[conservadas]


    # Sobrescribir df sin duplicados generales
    df = df.drop_duplicates(keep='first')
    print((f'✅ Filtro de duplicados generales aplicado. {df.shape[0]} filas y {df.shape[1]} columnas.'))
    # Sobrescribir df sin duplicados especificos
    df = df.drop_duplicates(subset=["id_property_id"], keep='first')
    print((f'✅ Filtro de duplicados específicos (property_id) aplicado. {df.shape[0]} filas y {df.shape[1]} columnas.'))

    return df

# Renombrar columnas por tipo
def renombrar_por_tipo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame y renombra columnas según su tipo, utilizando prefijos:
      - id_ para id_feats
      - date_ para date_feats
      - num_ para num_feats
      - cat_ para cat_feats
      - text_ para text_feats
      - geo_ para geo_feats
      - tgt_ para tgt_feats

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame cuyas columnas serán renombradas.

    Retorno:
    --------
    pandas.DataFrame
        El DataFrame con las columnas renombradas.
    """
    id_feats = []
    date_feats = []
    num_feats = ['aluminum_price', 'crude_oil_price', 'dollar_index', 'copper_price', 'hot_rolled_coil_price', 'coking_coal_price', 'iron_ore_price', 'usd_cny', 'sp500', 'vix', 'usd_mxn']
    cat_feats=[]
    text_feats = []
    geo_feats = []
    tgt_feats = ['rebar_price']

    def _make_map(cols, prefix):
        return {col: f"{prefix}{col}" for col in cols if col in df.columns}

    rename_map = {}
    rename_map.update(_make_map(id_feats, "id_"))
    rename_map.update(_make_map(date_feats, "date_"))
    rename_map.update(_make_map(num_feats, "num_"))
    rename_map.update(_make_map(cat_feats, "cat_"))
    rename_map.update(_make_map(text_feats, "text_"))
    rename_map.update(_make_map(geo_feats, "geo_"))
    rename_map.update(_make_map(tgt_feats, "tgt_"))

    return df.rename(columns=rename_map)

# Cargar datos
def cargar_datos():

    # Cambiar el directorio de trabajo al raíz del proyecto
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Cargar el dataset de precio de varilla corrugada
    df_target = pd.read_csv('data/Steel_Rebar_Futures_Historical_Data.csv')
    df_target["Date"] = pd.to_datetime(df_target["Date"])
    df_target = df_target.sort_values("Date").set_index("Date")

    # Nos quedamos solo con el precio de cierre
    df_target = df_target[["Price"]].rename(columns={"Price": "rebar_price"}) # USD/ton


    yf_tickers = {
    "HRC=F": "hot_rolled_coil_price",    # Precio de Acero laminado en caliente - Hot Rolled Coil Steel USD/ton; La varilla corrugada se produce a partir de acero largo
    "TIO=F": "iron_ore_price",           # Precio de Mineral de hierro - Iron Ore USD/ton; materia prima principal de acero
    "MTF=F": "coking_coal_price",        # Precio de Carbón coquizable (para alto horno). - Coking Coal USD/ton
    "HG=F": "copper_price",              # Precio del cobre - Copper USD/libra
    "CL=F": "crude_oil_price",           # Precio del petroleo - Crude Oil USD/barril
    "USDCNY=X": "usd_cny",               # Tipo de cambio USD / Yuan chino (CNY)  CNY/dolar americano
    "ALI=F": "aluminum_price",           # Precio de Aluminio USD/ton
    "DX-Y.NYB": "dollar_index",          # mide la fortaleza del dólar estadounidense  índice(base 100)
    "^VIX": "vix",                       # volatilidad esperada del S&P 500     índice(puntos de volatilidad)
    "^GSPC": "sp500"                     # índice bursátil de las 500 empresas más grandes de EE. UU
    }

    df_yf  = yf.download(
        list(yf_tickers.keys()),
        start="2016-01-01",
        auto_adjust=True,
        progress=False
        )

    df_yf = df_yf["Close"]  # solo precios de cierre

    df_yf = df_yf.rename(columns=yf_tickers) # renombre de columnas


    start = dt.datetime(2016, 1, 1)

    df_fx = pdr.DataReader(
        "DEXMXUS",  # Tipo de cambio USD/ MXN   MXN/ dolar americano
        "fred",
        start
    )

    df_fx.columns = ["usd_mxn"]

    df = pd.concat([df_target, df_yf, df_fx], axis=1).round(1)
    df = df.sort_index()
    #display(df)
    df = df.asfreq("B") # Fija frecuencia de días hábiles (CRÍTICO)
    #display(df)

    print(f'✅ Dataset cargado correctamente. {df.shape[0]} filas y {df.shape[1]} columnas.')

    print(
    f"Datos de entrada para producción desde {df.index.min().date()} "
    f"hasta {df.index.max().date()} "
    f"({len(df)} obs) "
    )
    
    return df

def pipeline_preprocesamiento():
    df = cargar_datos()
    df = renombrar_por_tipo(df)
    df = filtrar_por_completitud(df, umbral=80)
    df = filtro_unarias(df, umbral=0.9, mostrar_sesgo=False)
    df = feature_engineering(df) 
    df_live, df_train = imputar_missings(df)
        
    return df_live, df_train

def preparar_modelo(df):

    ## Columnas relevantes para el modelo
    x_cols = [c for c in df.columns if c.startswith(("num_", "tgt_rebar_price_lag"))]
    FEATURES = x_cols
    print("features usadas para entrenamiento en producción: ")
    for c in FEATURES:
        print(f"   - {c}")
    TARGET = "target_t_plus_1"

    X = df[FEATURES]
    y = df[TARGET]

    xgbr_model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
                )

    xgbr_model.fit(X, y)

    print(
        f"Datos para entrenamiento en producción desde {y.index.min().date()} "
        f"hasta {y.index.max().date()} "
        f"({len(y)} obs) "
    )

    return xgbr_model
