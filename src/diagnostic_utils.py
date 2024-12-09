import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf

def verify_preprocessing(scaled_data, raw_data, prediction_probabilities):
    """Verifica y muestra el preprocesamiento de datos"""
    st.write("🔍 Diagnóstico de Preprocesamiento:")

    # Comparación de datos
    comparison = pd.DataFrame({
        'Variable': ['tenure', 'age', 'income', 'ed', 'employ', 'address', 'reside'],
        'Original': [raw_data['tenure'], raw_data['age'], raw_data['income'],
                    raw_data['ed'], raw_data['employ'], raw_data['address'], 1.0],
        'Escalado': scaled_data[0][:7]
    })
    st.write("Comparación de valores originales vs escalados:")
    st.dataframe(comparison)

    # Probabilidades por categoría
    st.write("\nProbabilidades por categoría:")
    categories = ["Basic Service", "E-Service", "Plus Service", "Total Service"]
    probs_df = pd.DataFrame({
        'Categoría': categories,
        'Probabilidad': prediction_probabilities[0]
    })
    st.dataframe(probs_df)

def get_model_summary(model):
    """Genera un resumen del modelo"""
    st.write("🤖 Información del modelo:")

    for i, layer in enumerate(model.layers):
        st.write(f"\nCapa {i+1}:")
        st.write(f"- Nombre: {layer.name}")
        st.write(f"- Tipo: {type(layer).__name__}")