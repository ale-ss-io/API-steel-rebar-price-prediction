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

**Interpretación de solución para negocio**:

1. **Problema de regresión: estimación del monto de pago mensual (PAY_AMT4)**
   
   Para el problema de regresión se evaluaron tres modelos principales: Random Forest, XGBoost y LightGBM. La métrica objetivo fue R², dado que mide la proporción de variabilidad del monto pagado que el modelo es capaz de explicar.
   
   El modelo con mejor desempeño fue XGBoost, alcanzando un R² ≈ 0.82. Esto indica que el modelo es capaz de explicar alrededor del 82% de la variación del monto de pago mensual. Para negocio, este nivel de precisión permitiría anticipar montos esperados de pago, generar estrategias de cobranza diferenciadas y optimizar flujos financieros.

3. **Problema de clasificación: predicción de incumplimiento de pago (default.payment.next.month)**
   
   Para la clasificación se utilizaron los mismos algoritmos, ahora en su versión de clasificación. La métrica seleccionada fue el F1-score de la clase 1, ya que representa un equilibrio entre la capacidad de detectar casos de incumplimiento (recall) y la precisión al hacerlo (precision).
   
   El mejor modelo fue Random Forest, alcanzando un Recall de Clase 1 ≈ 54%. Esto significa que el modelo es capaz de identificar poco más de la mitad de los clientes que incurrirán en mora. Aunque el desempeño es razonable, aún existe margen de mejora si el objetivo del negocio es detectar más casos de riesgo de mora. 
