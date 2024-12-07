# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements y del proyecto
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código al contenedor
COPY src/ ./src
COPY models/ ./models

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV MONGODB_URI=mongodb://mongodb:27017/
ENV MONGODB_DATABASE=customer

# Dar permisos a los directorios
RUN chmod -R 755 /app/src
RUN chmod -R 755 /app/models

# Exponer el puerto para Streamlit
EXPOSE 8501

# Verificar la estructura de archivos
RUN echo "Contenido de /app:" && ls -la /app && \
    echo "Contenido de /app/src:" && ls -la /app/src && \
    echo "Contenido de /app/models:" && ls -la /app/models

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]