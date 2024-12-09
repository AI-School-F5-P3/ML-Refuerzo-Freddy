import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

def load_metrics():
    try:
        with open('../models/promotion_history.json', 'r') as f:
            metrics = [json.loads(line) for line in f]
        return pd.DataFrame(metrics)
    except:
        return pd.DataFrame()

def load_drift_data():
    try:
        with open('../models/drift_monitoring.json', 'r') as f:
            drift_data = [json.loads(line) for line in f]
        return drift_data
    except:
        return []

def main():
    st.title("MLOps Dashboard - Telecom Customer Prediction")

    # Métricas del modelo
    st.header("📊 Métricas del Modelo")
    metrics_df = load_metrics()

    if not metrics_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy Actual", f"{metrics_df['accuracy'].iloc[-1]:.4f}")
        with col2:
            st.metric("Mejora", f"{metrics_df['improvement'].iloc[-1]:+.4f}")

        # Gráfico de evolución
        fig = px.line(metrics_df, x='timestamp', y='accuracy',
                     title='Evolución del Accuracy')
        st.plotly_chart(fig)

    # Monitoreo de Drift
    st.header("🔄 Monitoreo de Data Drift")
    drift_data = load_drift_data()

    if drift_data:
        latest_drift = drift_data[-1]

        # Mostrar features con drift significativo
        significant_drift = {k: v for k, v in latest_drift['features'].items()
                           if v['significant']}

        if significant_drift:
            st.warning("Features con drift significativo detectado:")
            for feature, data in significant_drift.items():
                st.write(f"- {feature}: {data['drift']:.4f}")
        else:
            st.success("No se detectó drift significativo")

if __name__ == "__main__":
    main()