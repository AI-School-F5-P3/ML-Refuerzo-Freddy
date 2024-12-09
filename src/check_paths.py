import os

def check_model_paths():
    """Verifica las rutas de los modelos"""
    # Obtener ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Directorio actual: {current_dir}")

    # Ruta a la carpeta models
    models_dir = os.path.join(current_dir, '..', 'models')
    print(f"Directorio de modelos: {models_dir}")

    # Verificar si existe el directorio
    if os.path.exists(models_dir):
        print("✓ Directorio models encontrado")
        # Listar contenido
        files = os.listdir(models_dir)
        print("\nArchivos encontrados:")
        for file in files:
            print(f"- {file}")
    else:
        print("× Directorio models no encontrado")

if __name__ == "__main__":
    check_model_paths()