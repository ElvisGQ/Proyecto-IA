import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import PolynomialFeatures

# Load models and encoders
models = {
    "XGBoost": load('models/classification/xgboost_parking_model.pkl'),
    "Random Forest": load('models/classification/randomforest_parking_model.pkl'),
    "Logistic Regression": load('models/classification/logistic_parking_model.pkl'),  # 👈 added here
    "CatBoost": load('models/classification/catboost_parking_model.pkl')  # Added CatBoost model
}

le_dia = load('encoders/classification/encoder_dia.pkl')
le_clima = load('encoders/classification/encoder_clima.pkl')
le_estado = load('encoders/classification/encoder_estado.pkl')
poly = load('encoders/classification/poly_transformer.pkl')  # Assuming you saved this transformer when training
scaler = load('encoders/classification/scaler.pkl')  # 👈 You need to have saved this during training

# Sidebar inputs
st.sidebar.title("🔧 Configuración")
selected_model_name = st.sidebar.selectbox("Modelo a usar", list(models.keys()))
custom_order_dia = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
dia = st.sidebar.selectbox("Día", custom_order_dia)
hora = st.sidebar.slider("Hora (formato 24h)", min_value=8, max_value=21, value=14)
clima = st.sidebar.selectbox("Clima", le_clima.classes_)

# Main title
st.title("Visualizador de Estacionamiento")

# Encode inputs
dia_encoded = le_dia.transform([dia])[0]
clima_encoded = le_clima.transform([clima])[0]
model = models[selected_model_name]

# Predict for 20 spaces
predictions = []
for espacio in range(1, 21):
    input_df = pd.DataFrame([{
        'Día': dia_encoded,
        'Hora': hora,
        'Espacio': espacio,
        'Clima': clima_encoded
    }])

    # Apply Polynomial Features if using Logistic Regression
    if selected_model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)       # ✅ scale first
        input_df = poly.transform(input_scaled)          # ✅ then transform
    pred = model.predict(input_df)
    estado = le_estado.inverse_transform(pred)[0]
    predictions.append(estado)

# Count available
disponibles = sum(1 for estado in predictions if estado.lower() == 'disponible')

# CSS Styles
st.markdown("""
<style>
    .parking-lot {
        display: grid;
        grid-template-columns: repeat(5, 100px);
        gap: 10px;
        justify-content: center;
        margin-top: 20px;
    }
    .spot {
        height: 80px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
        color: white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .available {
        background-color: #4CAF50;
    }
    .occupied {
        background-color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Build grid layout
html = '<div class="parking-lot">'
for i, estado in enumerate(predictions):
    css_class = "available" if estado.lower() == "disponible" else "occupied"
    html += f'<div class="spot {css_class}">P{i+1}<br>{estado}</div>'
html += '</div>'

# Show layout
st.markdown("### Mapa del estacionamiento:")
st.markdown(html, unsafe_allow_html=True)

# Summary
st.markdown("---")
st.success(f" Espacios disponibles: **{disponibles}** de 20")
