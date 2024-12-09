import streamlit as st
import pandas as pd
import numpy as np
from database import init_mongodb
import plotly.express as px
from datetime import datetime
import os
import tensorflow as tf
from diagnostic_utils import verify_preprocessing, get_model_summary

# Configuración de la página
st.set_page_config(
    page_title="Telecom Customer Prediction",
    page_icon="📱",
    layout="wide"
)

# Función para escalar características
def scale_features(data):
    """Escala las características numéricas usando los parámetros del EDA"""
    numeric_cols = ['tenure', 'age', 'income', 'ed', 'employ', 'address', 'reside']

    scaling_params = {
        'tenure': {'mean': 35.0, 'std': 21.0},
        'age': {'mean': 41.0, 'std': 12.0},
        'income': {'mean': 63000.0, 'std': 25000.0},
        'ed': {'mean': 2.7, 'std': 1.1},
        'employ': {'mean': 5.0, 'std': 2.8},
        'address': {'mean': 3.7, 'std': 1.7},
        'reside': {'mean': 3.0, 'std': 1.0}
    }

    scaled_data = data.copy()
    for col in numeric_cols:
        if col in scaling_params:
            mean = scaling_params[col]['mean']
            std = scaling_params[col]['std']
            scaled_data[col] = (data[col] - mean) / std

    return scaled_data

def prepare_input_data(tenure, age, income, education, employ, address, gender, marital, region):
    """
    Prepara los datos de entrada con exactamente 19 características
    """
    # 1. Escalar características numéricas
    input_dict = {
        'tenure': float(tenure),
        'age': float(age),
        'income': float(income),
        'ed': float(education),
        'employ': float(employ),
        'address': float(address),
        'reside': 1.0
    }
    scaled_data = scale_features(pd.DataFrame([input_dict]))

    # Lista para todas las características
    features = []

    # 2. Características numéricas escaladas (7)
    for col in ['tenure', 'age', 'income', 'ed', 'employ', 'address', 'reside']:
        features.append(float(scaled_data[col].iloc[0]))

    # 3. Género (2)
    features.extend([
        1.0 if gender == "Male" else 0.0,    # male
        0.0 if gender == "Male" else 1.0     # female
    ])

    # 4. Estado civil (3)
    marital_encoding = {
        'Single': [1, 0, 0],    # single, married, divorced
        'Married': [0, 1, 0],
        'Divorced': [0, 0, 1]
    }
    features.extend(marital_encoding[marital])

    # 5. Región (4)
    region_encoding = {
        'North': [1, 0, 0, 0],  # north, south, east, west
        'South': [0, 1, 0, 0],
        'East': [0, 0, 1, 0],
        'West': [0, 0, 0, 1]
    }
    features.extend(region_encoding[region])

    # 6. Características engineered (3)
    income_scaled = float(scaled_data['income'].iloc[0])
    tenure_scaled = float(scaled_data['tenure'].iloc[0])
    age_scaled = float(scaled_data['age'].iloc[0])
    ed_scaled = float(scaled_data['ed'].iloc[0])

    features.extend([
        income_scaled / 2.0,                     # income_per_family
        tenure_scaled / age_scaled if age_scaled > 0 else 0.0,  # tenure_to_age_ratio
        income_scaled / ed_scaled if ed_scaled > 0 else 0.0     # income_per_education
    ])

    # Verificar número de características
    assert len(features) == 19, f"Error: Generadas {len(features)} características, se esperaban 19"

    # Convertir a numpy array
    return np.array([features], dtype=np.float32)

@st.cache_resource
def load_keras_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', 'models')

        final_model_path = os.path.join(models_dir, 'final_model.keras')
        if os.path.exists(final_model_path):
            print(f"Cargando modelo final: {final_model_path}")
            return tf.keras.models.load_model(final_model_path)

        best_model_path = os.path.join(models_dir, 'best_model.keras')
        if os.path.exists(best_model_path):
            print(f"Cargando mejor modelo: {best_model_path}")
            return tf.keras.models.load_model(best_model_path)

        raise FileNotFoundError("No se encontraron modelos válidos")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Inicializar MongoDB
db = init_mongodb()
if db is None:
    st.error("❌ No se pudo conectar a MongoDB")
    st.info("Verifica que MongoDB está corriendo en el puerto 27017")
    st.stop()

def main():
    st.title('📱 Telecom Customer Prediction')

    with st.sidebar:
        st.success("✅ Conectado a MongoDB")
        page = st.radio("Navegación", ["Nueva Predicción", "Dashboard", "Historial"])

    if page == "Nueva Predicción":
        show_prediction_page()
    elif page == "Dashboard":
        show_dashboard_page()
    else:
        show_history_page()

def show_prediction_page():
    st.header("🎯 Nueva Predicción")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("Tiempo como cliente (meses)", min_value=0)
            age = st.number_input("Edad", min_value=18, max_value=100, value=30)
            income = st.number_input("Ingresos", min_value=0.0, value=50000.0)

        with col2:
            education = st.selectbox("Nivel de educación",
                                   options=[1, 2, 3, 4, 5],
                                   format_func=lambda x: f"Nivel {x}")
            employ = st.number_input("Años de empleo", min_value=0)
            address = st.number_input("Años en dirección actual", min_value=0)

        with col3:
            gender = st.selectbox("Género", ["Male", "Female"])
            marital = st.selectbox("Estado civil", ["Single", "Married", "Divorced"])
            region = st.selectbox("Región", ["North", "South", "East", "West"])

        submitted = st.form_submit_button("Predecir")

        if submitted:
            model = load_keras_model()
            if model:
                try:
                    # Preparar datos y hacer predicción
                    input_data = prepare_input_data(
                        tenure, age, income, education, employ, address,
                        gender, marital, region
                    )

                    with st.spinner('Realizando predicción...'):
                        prediction = model.predict(input_data, verbose=0)

                    # Mostrar diagnóstico
                    with st.expander("Diagnóstico de datos", expanded=True):
                        st.write("Detalle de características generadas:")
                        feature_names = [
                            "tenure", "age", "income", "ed", "employ", "address", "reside",
                            "gender_male", "gender_female",
                            "marital_single", "marital_married", "marital_divorced",
                            "region_north", "region_south", "region_east", "region_west",
                            "income_per_family", "tenure_to_age_ratio", "income_per_education"
                        ]

                        st.write("Forma del tensor de entrada:", input_data.shape)
                        st.write("\nValores por característica:")
                        for i, (name, value) in enumerate(zip(feature_names, input_data[0])):
                            st.write(f"{i+1}. {name}: {value:.4f}")

                        st.write("\nProbabilidades por categoría:")
                        for i, prob in enumerate(prediction[0]):
                            st.write(f"Categoría {i+1}: {prob:.4%}")

                    # Procesar resultados
                    predicted_class = np.argmax(prediction[0])
                    probability = float(np.max(prediction[0]))

                    # Guardar en MongoDB
                    customer_data = {
                        'tenure': tenure,
                        'age': age,
                        'income': income,
                        'education_level': education,
                        'employment_years': employ,
                        'address_years': address,
                        'gender': gender,
                        'marital_status': marital,
                        'region': region,
                        'predicted_category': int(predicted_class + 1),
                        'prediction_probability': probability,
                        'all_probabilities': prediction[0].tolist()
                    }

                    if db.insert_customer(customer_data):
                        st.success(f"📊 Categoría predicha: {predicted_class + 1}")
                        st.info(f"Probabilidad: {probability:.2%}")

                        categories_desc = {
                            1: "Basic Service - Cliente con servicios básicos",
                            2: "E-Service - Cliente con servicios electrónicos",
                            3: "Plus Service - Cliente con servicios premium",
                            4: "Total Service - Cliente con servicios completos"
                        }
                        st.write(categories_desc[predicted_class + 1])
                    else:
                        st.error("Error al guardar la predicción")

                except Exception as e:
                    st.error(f"Error durante la predicción: {str(e)}")
                    st.error("Detalles del error para debugging:")
                    st.code(str(e))

def show_dashboard_page():
    st.header("📊 Dashboard")

    stats = db.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Clientes", stats['total_customers'])

    if stats['total_customers'] > 0:
        col1, col2 = st.columns(2)

        with col1:
            if stats['predictions_by_category']:
                df_categories = pd.DataFrame(stats['predictions_by_category'])
                fig = px.pie(df_categories, values='count', names='_id',
                           title='Distribución de Categorías')
                st.plotly_chart(fig)

        with col2:
            if stats['customers_by_region']:
                df_regions = pd.DataFrame(stats['customers_by_region'])
                fig = px.bar(df_regions, x='_id', y='count',
                           title='Clientes por Región')
                st.plotly_chart(fig)

def show_history_page():
    st.header("📜 Historial de Predicciones")

    predictions = db.get_recent_predictions(limit=10)

    if predictions:
        for pred in predictions:
            with st.expander(f"Cliente {pred['age']} años - {pred['region']} - {pred['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("📋 Datos del cliente:")
                    st.write(f"- Edad: {pred['age']}")
                    st.write(f"- Región: {pred['region']}")
                    st.write(f"- Género: {pred['gender']}")
                    st.write(f"- Estado Civil: {pred['marital_status']}")
                    st.write(f"- Ingresos: ${pred['income']:,.2f}")
                with col2:
                    st.write("🎯 Resultado:")
                    st.write(f"- Categoría: {pred['predicted_category']}")
                    st.write(f"- Probabilidad: {pred['prediction_probability']:.2%}")
                    if 'all_probabilities' in pred:
                        st.write("Probabilidades por categoría:")
                        categories = ["Basic", "E-Service", "Plus", "Total"]
                        for cat, prob in zip(categories, pred['all_probabilities']):
                            st.write(f"- {cat}: {prob:.2%}")
                    st.write(f"- Tiempo como cliente: {pred['tenure']} meses")
                    st.write(f"- Nivel de educación: {pred['education_level']}")
    else:
        st.info("No hay predicciones registradas")

if __name__ == "__main__":
    main()