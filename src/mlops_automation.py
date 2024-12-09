import mlflow
import schedule
import time
from datetime import datetime
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from rich.console import Console
from sklearn.metrics import classification_report


console = Console()

class MLOpsAutomation:
    def __init__(self):
        # Configuración de rutas y archivos
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_path = os.path.join(self.root_dir, '../data/proc_escalado.csv')

        self.models = {
            "Best Model": os.path.join(self.root_dir, '../models/best_model.keras'),
            "Final Model": os.path.join(self.root_dir, '../models/final_model.keras'),
            "Best Model 01": os.path.join(self.root_dir, '../models/best_model01_20241209_163533.keras'),
            "Final Model 01": os.path.join(self.root_dir, '../models/final_model01_20241209_163533.keras')
        }

        self.logs_dir = os.path.join(self.root_dir, '../logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.logs_dir, 'mlops_automation.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self._check_files()
        mlflow.set_experiment("telecom_customer_mlops")
        console.print("[green]✓[/green] MLOps Automation initialized")

    def _check_files(self):
        self.files_status = {
            'Raw Data': os.path.exists(self.raw_data_path),
            **{name: os.path.exists(path) for name, path in self.models.items()}
        }
        console.print("\n[bold]Estado de archivos:[/bold]")
        for name, exists in self.files_status.items():
            status = "[green]✓[/green]" if exists else "[red]✗[/red]"
            path = self.models.get(name, self.raw_data_path)
            console.print(f"{status} {name}: {path}")

    def prepare_data(self):
        """Preparar datos"""
        try:
            df = pd.read_csv(self.raw_data_path)
            categorical_columns = ['age_segment', 'income_segment', 'tenure_segment']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = pd.Categorical(df[col]).codes
            X = df.drop('custcat', axis=1)
            y = df['custcat'] - 1
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            console.print(f"[red]Error en preparación de datos: {str(e)}[/red]")
            return None, None, None, None

    def evaluate_model(self, model_name, model_path, X_test, y_test, nested_run=False):
        """Evaluar un modelo específico"""
        try:
            with mlflow.start_run(nested=nested_run):
                model = load_model(model_path)
                predictions = model.predict(X_test)
                pred_classes = np.argmax(predictions, axis=1)
                results = classification_report(y_test, pred_classes, output_dict=True)
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("accuracy", results['accuracy'])
                mlflow.log_metric("weighted_f1", results['weighted avg']['f1-score'])
                return results
        except Exception as e:
            console.print(f"[red]Error evaluando {model_name}: {str(e)}[/red]")
            return None

    def monitor_metrics(self):
        """Monitorear métricas de todos los modelos"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        for model_name, model_path in self.models.items():
            with mlflow.start_run(run_name=f"{model_name}_monitoring"):
                self.evaluate_model(model_name, model_path, X_test, y_test, nested_run=True)

    def check_model_drift(self):
        """Verificar drift del modelo"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        for model_name, model_path in self.models.items():
            with mlflow.start_run(run_name=f"{model_name}_drift_check"):
                self.evaluate_model(model_name, model_path, X_test, y_test, nested_run=True)

    def run_automation(self):
        """Ejecutar automatización MLOps"""
        console.print("\n[bold blue]Iniciando automatización MLOps[/bold blue]")
        if not all(self.files_status.values()):
            console.print("\n[red]⚠️ Advertencia: Faltan algunos archivos necesarios[/red]")
            return
        self.monitor_metrics()
        self.check_model_drift()
        schedule.every().hour.do(self.monitor_metrics)
        schedule.every().day.at("00:00").do(self.check_model_drift)
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            console.print("[green]✓ MLOps automation detenido correctamente[/green]")

if __name__ == "__main__":
    mlops = MLOpsAutomation()
    mlops.run_automation()
