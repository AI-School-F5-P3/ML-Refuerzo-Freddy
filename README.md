Proyecto de Clasificación de Clientes de Telecomunicaciones
Descripción del Proyecto

Este proyecto tiene como objetivo construir un modelo de aprendizaje supervisado para clasificar a los clientes de una empresa de telecomunicaciones en diferentes categorías. Utilizando técnicas avanzadas de machine learning, análisis de datos y automatización mediante MLOps, el proyecto busca identificar patrones clave para mejorar la experiencia del cliente y aumentar la retención.

Estructura del Proyecto
1. Carpeta data
Contiene los datasets utilizados en el proyecto:

teleCust1000t.csv: Dataset original.
proc_escalado.csv: Dataset preprocesado y escalado, listo para el entrenamiento.

2. Carpeta logs
mlops_automation.log: Registro de las actividades automatizadas realizadas por el script de MLOps.
Archivos de comparación de modelos generados:
model_comparison_YYYYMMDD_HHmmss.csv: Comparación de métricas entre modelos evaluados.

3. Carpeta models
best_model.keras: Modelo con los mejores resultados basado en redes neuronales.
final_model.keras: Modelo ajustado con hiperparámetros optimizados.
Checkpoints: Pesos intermedios y modelos generados durante los entrenamientos.

4. Carpeta notebook
Contiene notebooks para análisis exploratorio y experimentación:

EDA-FINAL.ipynb: Análisis exploratorio de datos (EDA) detallado.
Otros notebooks para experimentos con SMOTE-Tomek y generación de gráficos.

5. Carpeta src
Scripts principales del proyecto:

app.py: Implementación de una aplicación Streamlit para realizar predicciones.
retrain_model.py: Script para reentrenar modelos.
mlops_automation.py: Automatización del reentrenamiento y monitorización de modelos usando MLOps.
models-camparacion-mlflow.py: Comparación de métricas entre modelos usando MLflow.
database.py: Conexión con bases de datos relacionadas con el proyecto.

6. Archivos de configuración
Dockerfile: Archivo para contenerizar el proyecto.
docker-compose.yml: Configuración para ejecutar servicios como MLflow.
.env: Variables de entorno necesarias para ejecutar el proyecto.
Resultados del Proyecto

Modelos Evaluados

Modelo	Accuracy	F1 ponderado	Duración (ms)
Final Model 01	0.375	0.339	326
Best Model 01	0.400	0.360	350

Conclusión
El modelo Best Model 01 es recomendado para su despliegue debido a su mejor rendimiento en precisión y métricas de clasificación.

Cómo Ejecutar el Proyecto
1. Configurar el Entorno
Clona este repositorio:

git clone <URL-del-repositorio>
cd telecom_customer
Crea un entorno virtual:

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Analizar los Datos
Ejecuta el análisis exploratorio:

jupyter notebook notebook/EDA-FINAL.ipynb

3. Reentrenar el Modelo
Reentrena el modelo y guarda los resultados:

python src/retrain_model.py

4. Automatización con MLOps
Ejecuta el script de automatización:
python src/mlops_automation.py

5. Visualización con MLflow
Inicia MLflow:

mlflow ui
Accede a http://localhost:5000 para explorar los experimentos y modelos registrados.

6. Despliegue de la Aplicación
Lanza la aplicación Streamlit para realizar predicciones:

streamlit run src/app.py

Autor
Freddy Matareno