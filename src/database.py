from pymongo import MongoClient
from datetime import datetime
import streamlit as st

class MongoDB:
    def __init__(self):
        # Usar el nombre del servicio de MongoDB en Docker
        self.client = MongoClient('mongodb://mongodb:27017/')
        self.db = self.client['customer']
        self.customers = self.db['customers']

    def insert_customer(self, customer_data):
        try:
            customer_data['timestamp'] = datetime.utcnow()
            result = self.customers.insert_one(customer_data)
            return result.inserted_id
        except Exception as e:
            st.error(f"Error al insertar datos: {str(e)}")
            return None

    def get_recent_predictions(self, limit=5):
        try:
            return list(self.customers.find().sort('timestamp', -1).limit(limit))
        except Exception as e:
            st.error(f"Error al obtener predicciones recientes: {str(e)}")
            return []

    def get_stats(self):
        try:
            return {
                'total_customers': self.customers.count_documents({}),
                'predictions_by_category': list(self.customers.aggregate([
                    {'$group': {
                        '_id': '$predicted_category',
                        'count': {'$sum': 1}
                    }}
                ])),
                'customers_by_region': list(self.customers.aggregate([
                    {'$group': {
                        '_id': '$region',
                        'count': {'$sum': 1}
                    }}
                ]))
            }
        except Exception as e:
            st.error(f"Error al obtener estadísticas: {str(e)}")
            return {
                'total_customers': 0,
                'predictions_by_category': [],
                'customers_by_region': []
            }

@st.cache_resource
def init_mongodb():
    try:
        db = MongoDB()
        db.client.server_info()
        return db
    except Exception as e:
        st.error(f"Error al conectar con MongoDB: {str(e)}")
        return None