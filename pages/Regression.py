# pages/Prediction.py
import streamlit as st
import pandas as pd
from joblib import load

# Load regression models and preprocessor
models = {
    "Linear Regression": load('models/regression/linear_regression_model.pkl'),
    "Random Forest": load('models//regression/random_forest_model.pkl')
}

encoder_clima = load('encoders/regression/clima_encoder.pkl')  # OneHotEncoder
custom_order_dia = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']

# Sidebar Inputs
st.sidebar.title("🔧 Configuración")
selected_model_name = st.sidebar.selectbox("Modelo de regresión", list(models.keys()))
dia = st.sidebar.selectbox("Día", custom_order_dia)
hora = st.sidebar.slider("Hora (formato 24h)", min_value=8, max_value=21, value=14)
clima = st.sidebar.selectbox("Clima", encoder_clima.categories_[0])

# Main title
st.title("Predicción de Espacios Disponibles")

# Encode categorical inputs
dia_encoded = custom_order_dia.index(dia)  # simple label encoding (same as during training)
clima_encoded = encoder_clima.transform(pd.DataFrame([[clima]], columns=['clima']))

# Assemble feature vector
X_input = pd.DataFrame([[dia_encoded, hora]], columns=['Día', 'Hora'])
X_full = pd.concat([X_input.reset_index(drop=True), pd.DataFrame(clima_encoded, columns=encoder_clima.get_feature_names_out(['clima']))], axis=1)

# Predict
model = models[selected_model_name]
prediction = model.predict(X_full)[0]

# Format result
espacios_disponibles = max(0, min(round(prediction), 20))  # Clamp between 0 and 20

# Output
# Create summary DataFrame
summary_df = pd.DataFrame({
    "Modelo": [selected_model_name],
    "Día": [dia],
    "Hora": [f"{hora}:00"],
    "Clima": [clima]
})

# Display title and table
st.markdown("### Resumen de la predicción")
st.dataframe(summary_df, use_container_width=True)

st.success(f" Espacios disponibles: **{espacios_disponibles}** de 20")