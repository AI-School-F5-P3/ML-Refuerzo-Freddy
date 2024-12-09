import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import datetime

def prepare_data():
    """Usar la misma preparación de datos que la red neuronal original"""
    try:
        # Ruta actualizada para el archivo de datos
        df = pd.read_csv('data/proc_escalado.csv')  # Ruta relativa a la raíz del proyecto
        categorical_columns = ['age_segment', 'income_segment', 'tenure_segment']

        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes

        X = df.drop('custcat', axis=1)
        y = df['custcat'] - 1  # Ajustar clases para empezar en 0

        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error en preparación de datos: {str(e)}")
        raise

def evaluate_models():
    """Evaluar y comparar modelos usando MLflow"""
    mlflow.set_experiment("telecom_customer_comparison")

    X_train, X_test, y_train, y_test = prepare_data()

    models_results = {}

    # 1. Evaluar best_model.keras
    with mlflow.start_run(run_name="best_model_neural_network"):
        try:
            nn_model = load_model('models/best_model.keras')  # Ruta actualizada
            y_pred_nn_best = np.argmax(nn_model.predict(X_test), axis=1)
            best_model_report = classification_report(y_test, y_pred_nn_best, output_dict=True)

            mlflow.log_metric("accuracy", best_model_report['accuracy'])
            mlflow.log_metric("weighted_f1", best_model_report['weighted avg']['f1-score'])
            models_results["Best Neural Network"] = best_model_report

            print("\nBest Model - Neural Network Results:")
            print(f"Accuracy: {best_model_report['accuracy']:.4f}")
            print(f"Weighted F1: {best_model_report['weighted avg']['f1-score']:.4f}")
        except Exception as e:
            print(f"Error evaluando best_model.keras: {str(e)}")

    # 2. Evaluar final_model.keras
    with mlflow.start_run(run_name="final_model_neural_network"):
        try:
            nn_model = load_model('models/final_model.keras')  # Ruta actualizada
            y_pred_nn_final = np.argmax(nn_model.predict(X_test), axis=1)
            final_model_report = classification_report(y_test, y_pred_nn_final, output_dict=True)

            mlflow.log_metric("accuracy", final_model_report['accuracy'])
            mlflow.log_metric("weighted_f1", final_model_report['weighted avg']['f1-score'])
            models_results["Final Neural Network"] = final_model_report

            print("\nFinal Model - Neural Network Results:")
            print(f"Accuracy: {final_model_report['accuracy']:.4f}")
            print(f"Weighted F1: {final_model_report['weighted avg']['f1-score']:.4f}")
        except Exception as e:
            print(f"Error evaluando final_model.keras: {str(e)}")

    # 3. Evaluar nuevo_modelo1.keras
    with mlflow.start_run(run_name="nuevo_modelo1_neural_network"):
        try:
            nuevo_modelo1 = load_model('models/best_model01_20241209_163533.keras')  # Ruta actualizada
            y_pred_nuevo1 = np.argmax(nuevo_modelo1.predict(X_test), axis=1)
            nuevo_modelo1_report = classification_report(y_test, y_pred_nuevo1, output_dict=True)

            mlflow.log_metric("accuracy", nuevo_modelo1_report['accuracy'])
            mlflow.log_metric("weighted_f1", nuevo_modelo1_report['weighted avg']['f1-score'])
            models_results["Nuevo Modelo 1"] = nuevo_modelo1_report

            print("\nNuevo Modelo 1 - Neural Network Results:")
            print(f"Accuracy: {nuevo_modelo1_report['accuracy']:.4f}")
            print(f"Weighted F1: {nuevo_modelo1_report['weighted avg']['f1-score']:.4f}")
        except Exception as e:
            print(f"Error evaluando nuevo_modelo1.keras: {str(e)}")

    # 4. Evaluar nuevo_modelo2.keras
    with mlflow.start_run(run_name="nuevo_modelo2_neural_network"):
        try:
            nuevo_modelo2 = load_model('models/final_model01_20241209_163533.keras')  # Ruta actualizada
            y_pred_nuevo2 = np.argmax(nuevo_modelo2.predict(X_test), axis=1)
            nuevo_modelo2_report = classification_report(y_test, y_pred_nuevo2, output_dict=True)

            mlflow.log_metric("accuracy", nuevo_modelo2_report['accuracy'])
            mlflow.log_metric("weighted_f1", nuevo_modelo2_report['weighted avg']['f1-score'])
            models_results["Nuevo Modelo 2"] = nuevo_modelo2_report

            print("\nNuevo Modelo 2 - Neural Network Results:")
            print(f"Accuracy: {nuevo_modelo2_report['accuracy']:.4f}")
            print(f"Weighted F1: {nuevo_modelo2_report['weighted avg']['f1-score']:.4f}")
        except Exception as e:
            print(f"Error evaluando nuevo_modelo2.keras: {str(e)}")

    # Crear tabla comparativa
    comparison = pd.DataFrame({
        'Modelo': [],
        'Accuracy': [],
        'Weighted F1': []
    })

    for model_name, report in models_results.items():
        new_row = {
            'Modelo': model_name,
            'Accuracy': report['accuracy'],
            'Weighted F1': report['weighted avg']['f1-score']
        }
        comparison = pd.concat([comparison, pd.DataFrame([new_row])], ignore_index=True)

    print("\nComparación final de modelos:")
    print(comparison)

    # Guardar resultados
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison.to_csv(f'logs/model_comparison_{timestamp}.csv', index=False)  # Ruta actualizada

    return comparison

if __name__ == "__main__":
    evaluate_models()