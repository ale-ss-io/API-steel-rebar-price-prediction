## Data Science Project: Modelado predictivo para detección de incumplimiento de pago (*default*) en clientes de tarjeta de crédito (clasificación), así como su monto de pago mensual (regresión) en el conjunto de datos 'Default of Credit Card Clients'.
________________________________________
**Objetivo**

El objetivo de este proyecto es, dado el conjunto de datos 'Default of Credit Card Clients', desarrollar modelos para predecir:
1) La cantidad de pago mensual (dólares NT) en Junio, 2005 por parte de un cliente.
2) Si o no un cliente cae en incumplimiento de pago (*default*) para Octubre 2005.
 ________________________________________

**Diccionario**
| Variable                       | Descripción                                                                                                                                                     |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ID**                         | Identificador de cada cliente                                                                                                                                   |
| **LIMIT_BAL**                  | Monto de crédito otorgado en dólares NT (incluye crédito individual y familiar/suplementario)                                                                   |
| **SEX**                        | Género (1 = hombre, 2 = mujer)                                                                                                                                  |
| **EDUCATION**                  | Nivel educativo (1 = posgrado, 2 = universidad, 3 = preparatoria, 4 = otros, 5 = desconocido, 6 = desconocido)                                                  |
| **MARRIAGE**                   | Estado civil (1 = casado, 2 = soltero, 3 = otros)                                                                                                               |
| **AGE**                        | Edad en años                                                                                                                                                    |
| **PAY_0**                      | Estado de pago en septiembre de 2005 (-1 = pago puntual, 1 = retraso de 1 mes, 2 = retraso de 2 meses, …, 8 = retraso de 8 meses, 9 = retraso de 9 meses o más) |
| **PAY_2**                      | Estado de pago en agosto de 2005 (misma escala que PAY_0)                                                                                                       |
| **PAY_3**                      | Estado de pago en julio de 2005 (misma escala que PAY_0)                                                                                                        |
| **PAY_4**                      | Estado de pago en junio de 2005 (misma escala que PAY_0)                                                                                                        |
| **PAY_5**                      | Estado de pago en mayo de 2005 (misma escala que PAY_0)                                                                                                         |
| **PAY_6**                      | Estado de pago en abril de 2005 (misma escala que PAY_0)                                                                                                        |
| **BILL_AMT1**                  | Monto del estado de cuenta en septiembre de 2005 (dólares NT)                                                                                                   |
| **BILL_AMT2**                  | Monto del estado de cuenta en agosto de 2005 (dólares NT)                                                                                                       |
| **BILL_AMT3**                  | Monto del estado de cuenta en julio de 2005 (dólares NT)                                                                                                        |
| **BILL_AMT4**                  | Monto del estado de cuenta en junio de 2005 (dólares NT)                                                                                                        |
| **BILL_AMT5**                  | Monto del estado de cuenta en mayo de 2005 (dólares NT)                                                                                                         |
| **BILL_AMT6**                  | Monto del estado de cuenta en abril de 2005 (dólares NT)                                                                                                        |
| **PAY_AMT1**                   | Monto del pago realizado en septiembre de 2005 (dólares NT)                                                                                                     |
| **PAY_AMT2**                   | Monto del pago realizado en agosto de 2005 (dólares NT)                                                                                                         |
| **PAY_AMT3**                   | Monto del pago realizado en julio de 2005 (dólares NT)                                                                                                          |
| **PAY_AMT4**                   | Monto del pago realizado en junio de 2005 (dólares NT)                                                                                                          |
| **PAY_AMT5**                   | Monto del pago realizado en mayo de 2005 (dólares NT)                                                                                                           |
| **PAY_AMT6**                   | Monto del pago realizado en abril de 2005 (dólares NT)                                                                                                          |
| **default.payment.next.month** | Incumplimiento de pago el próximo mes (1 = sí, 0 = no)                                                                                                          |

________________________________________
**Solución**

Nuestra solución consistió principalmente en:

1. Calidad de datos, analisis de datos exploratorio, pre-procesamiento de datos, entrenamiento y evaluación de varios modelos ML para:
   
   **Clasificación**: predecir la variable categórica **default.payment.next.month** y
   
   **Regresión**: predecir la variable númerica **PAY_AMT4**.

La estructura de este proyecto es:

```text

default-credit-card-clients/                         # Carpeta de proyecto
├── .gitignore
├── LICENSE
├── model_develop_credit_card_clients.ipynb          # Script de solución final
├── model_develop_credit_card_clients_develop.ipynb  # Script de desarrollo
├── PPT_Resumen.pptx                                 # Presentación de solución propuesta
├── README.md
├── Respuestas_a_Desarrollo_de_Solucion.docx         # Archivo Word de Respuestas a preguntas de cliente de este proyecto 
└── data/
    └── default_of_credit_card_clients.csv           # Archivo CSV de conjunto de datos 'Default of Credit Card Clients'

```

________________________________________
**Ejecución de Solución**

**Análisis**:
1. Abrir una terminal de Anaconda Prompt e ir a la carpeta donde descargaste este proyecto:
   ```bash
   cd your_local_root\default-credit-card-clients-regression-classification
2. Crear y activar el entorno virtual
   ```bash
   conda create -n default_of_credit_card_clients_env python=3.11.7 -y
   conda activate default_of_credit_card_clients_env
3. Instalar dependencias
   ```bash
   pip install -r requirements.txt
4. Navegar a 
   your_local_root\default-credit-card-clients-regression-classification
   y abrir el archivo
   model_develop_credit_card_clients.ipynb en su IDE prefererido, por ejemplo VSCode.

5. Ejecutar el script, indicando adecuadamente el environment previamente creado.

**Interpretación de solución para negocio**:

1. **Problema de regresión: estimación del monto de pago mensual (PAY_AMT4)**
   
   Para el problema de regresión se evaluaron tres modelos principales: Random Forest, XGBoost y LightGBM. La métrica objetivo fue R², dado que mide la proporción de variabilidad del monto pagado que el modelo es capaz de explicar.
   
   El modelo con mejor desempeño fue XGBoost, alcanzando un R² ≈ 0.82. Esto indica que el modelo es capaz de explicar alrededor del 82% de la variación del monto de pago mensual. Para negocio, este nivel de precisión permitiría anticipar montos esperados de pago, generar estrategias de cobranza diferenciadas y optimizar flujos financieros.

3. **Problema de clasificación: predicción de incumplimiento de pago (default.payment.next.month)**
   
   Para la clasificación se utilizaron los mismos algoritmos, ahora en su versión de clasificación. La métrica seleccionada fue el F1-score de la clase 1, ya que representa un equilibrio entre la capacidad de detectar casos de incumplimiento (recall) y la precisión al hacerlo (precision).
   
   El mejor modelo fue Random Forest, alcanzando un Recall de Clase 1 ≈ 54%. Esto significa que el modelo es capaz de identificar poco más de la mitad de los clientes que incurrirán en mora. Aunque el desempeño es razonable, aún existe margen de mejora si el objetivo del negocio es detectar más casos de riesgo de mora. 
