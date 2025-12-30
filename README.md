## Data Science Project: Modelado predictivo para predicción de precio de cierre (regresión) del día siguiente para la varilla corrugada usando una base de datos compuesta por variables financieras y del mercado.
________________________________________
**Objetivo**

El objetivo de este proyecto es desarrollar y desplegar una API REST que pueda:
1) Predecir el precio (USD/ton) del día siguiente de la varilla corrugada.
 ________________________________________
**Base de datos**

Para este proyecto decidimos usar tres conjunto de datos:
1) Precio de la varilla corrugada: Obtenida vía descarga manual en https://mx.investing.com/commodities/steel-rebar-historical-data
2) Yahoo Finance: Obtenida vía API de python
3) FRED: Obtenida vía API de python 

| Variable                  | Descripción                                                           | Fuente de datos                         | Descarga_RealTime |
| ------------------------- | --------------------------------------------------------------------- | --------------------------------------- | ----------------- |
| **rebar_price**           | Precio de varilla corrugada en USD/ton                                | Steel_Rebar_Futures_Historical_Data.csv | No                |
| **hot_rolled_coil_price** | Precio del acero laminado en caliente (Hot Rolled Coil) en USD/ton    | Yahoo Finance (HRC=F)                   | Sí                |
| **iron_ore_price**        | Precio del mineral de hierro (Iron Ore) en USD/ton                    | Yahoo Finance (TIO=F)                   | Sí                |
| **coking_coal_price**     | Precio del carbón coquizable (Coking Coal) para alto horno en USD/ton | Yahoo Finance (MTF=F)                   | Sí                |
| **copper_price**          | Precio del cobre en USD/libra                                         | Yahoo Finance (HG=F)                    | Sí                |
| **crude_oil_price**       | Precio del petróleo crudo en USD/barril                               | Yahoo Finance (CL=F)                    | Sí                |
| **usd_cny**               | Tipo de cambio USD/CNY (yuan chino por dólar estadounidense)          | Yahoo Finance (USDCNY=X)                | Sí                |
| **aluminum_price**        | Precio del aluminio en USD/ton                                        | Yahoo Finance (ALI=F)                   | Sí                |
| **dollar_index**          | Índice de fortaleza del dólar estadounidense (base 100)               | Yahoo Finance (DX-Y.NYB)                | Sí                |
| **vix**                   | Volatilidad esperada del S&P 500 (índice de puntos de volatilidad)    | Yahoo Finance (^VIX)                    | Sí                |
| **sp500**                 | Índice bursátil de las 500 empresas más grandes de EE. UU.            | Yahoo Finance (^GSPC)                   | Sí                |
| **usd_mxn**               | Tipo de cambio USD/MXN (peso mexicano por dólar estadounidense)       | FRED (DEXMXUS)                          | Sí                |




________________________________________
**Solución**

Nuestra solución consistió principalmente en:

1. Calidad de datos, analisis de datos exploratorio, pre-procesamiento de datos, entrenamiento y evaluación del modelo XGBRegressor:
   
   **Regresión**: predecir la variable de precio de varilla corrugada del siguiente día **target_t_plus_1**.

La estructura de este proyecto es:

```text

API-steel-rebar-price-prediction                     # Carpeta de proyecto
│   model_develop_steel_rebar.ipynb                  # Jupyter Notebook de análisis
│   README.md
│   requirements.txt
│   Reto_DS.pdf
│
├───appFastAPI
│   │   app.py                                       # Python Script de API REST
│   │   model.py                                     # Python Script de funciones en producción 
│   │   requirements.txt
│   │   __init__.py
│   │
│   └───__pycache__
│           app.cpython-313.pyc
│           model.cpython-313.pyc
│
└───data
        Steel_Rebar_Futures_Historical_Data.csv      # Base de datos en archivo .csv

```

________________________________________
**Ejecución de Solución**

**Análisis**:
1. Abrir una terminal de Anaconda Prompt e ir a la carpeta donde descargaste este proyecto:
   ```bash
   cd your_local_root\API-steel-rebar-price-prediction
2. Crear y activar el entorno virtual
   ```bash
   conda create -n API-steel-rebar-price-prediction_env python=3.11.4 -y
   conda activate API-steel-rebar-price-prediction_env
3. Instalar dependencias
   ```bash
   pip install -r requirements.txt
4. Navegar a 
   your_local_root\API-steel-rebar-price-prediction
   y abrir el archivo
   model_develop_steel_rebar.ipynb en su IDE prefererido, por ejemplo VSCode.

5. Ejecutar el script, indicando adecuadamente el environment previamente creado.

**API**:
1. Abrir una terminal de Anaconda Prompt e ir a la carpeta donde descargaste este proyecto:
   ```bash
   cd your_local_root\API-steel-rebar-price-prediction
2. Crear y activar el entorno virtual
   ```bash
   conda create -n API-steel-rebar-price-prediction_env python=3.11.4 -y
   conda activate API-steel-rebar-price-prediction_env
3. Instalar dependencias
   ```bash
   pip install -r requirements.txt
4. Ejecutar el servidor FastAPI
   uvicorn appFastAPI.app:app --reload
5. Abrir la documentación interactiva (Ver el link en la terminal de Anaconda Prompt) en un navegador web
   Ejemplo: http://127.0.0.1:8000/docs
6. En la API app, seleccionar el endpoint GET, inglesar el API_KEY = deacero-2025 y verá la predicción del precio de la varilla corrugada para el siguiente día:
   
  "prediction_date": "2025-12-31",
  "predicted_price_usd_per_ton": 596.79,
  "currency": "USD",
  "unit": "metric ton",
  "model_confidence": 0.8,
  "timestamp": "2025-12-30T17:23:02.194507Z"
   


**Interpretación de solución para negocio**:

Este proyecto proporciona un modelo predictivo de precios de la varilla corrugada en USD/ton, considerando factores macroeconómicos y materias primas. La interpretación clave para negocio es:

Precio de varilla (rebar_price): Permite anticipar tendencias de costos para planificación de compras y presupuestos de construcción.

Factores de influencia: Precios de acero laminado, mineral de hierro, carbón coquizable, aluminio y cobre; tipo de cambio USD/CNY y USD/MXN; volatilidad del S&P 500 (VIX). Esto ayuda a identificar qué variables macroeconómicas impactan más el precio.

Toma de decisiones: Con predicciones diarias, empresas pueden optimizar inventarios, negociar contratos y gestionar riesgos financieros.

Limitaciones: Algunos datos históricos se cargan desde CSV (Steel_Rebar_Futures_Historical_Data.csv) y no se actualizan automáticamente; otros (Yahoo Finance, FRED) sí se descargan en tiempo real, lo que permite predicciones más recientes y confiables.

Resumen: La herramienta soporta decisiones estratégicas y operativas en compras y planificación financiera, destacando riesgos y oportunidades en el mercado de la varilla corrugada.
