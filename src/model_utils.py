import os
from glob import glob
import tensorflow as tf

def find_latest_model():
    """
    Encuentra el modelo final más reciente en la carpeta models
    """
    try:
        # Buscar todos los modelos finales
        model_path = os.path.join('..', 'models', 'final_model.keras')
        models = glob(model_path)

        if not models:
            # Si no encuentra modelos finales, buscar best_model
            model_path = os.path.join('..', 'models', 'best_model.keras')
            models = glob(model_path)

        if not models:
            return None

        # Ordenar por fecha de modificación y tomar el más reciente
        latest_model = max(models, key=os.path.getctime)
        return latest_model
    except Exception as e:
        print(f"Error buscando el modelo: {str(e)}")
        return None

def load_model_safe():
    """
    Carga el modelo de manera segura con manejo de errores
    """
    try:
        model_path = find_latest_model()
        if model_path is None:
            raise FileNotFoundError("No se encontró ningún modelo .keras en la carpeta models")

        print(f"Cargando modelo: {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error cargando el modelo: {str(e)}")
        return None