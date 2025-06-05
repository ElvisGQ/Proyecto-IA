# pages/Regression.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Configuración de estilo ------------------
sns.set_style("darkgrid")
plt.rcParams.update({"figure.autolayout": True})

# ------------------ Cargar modelos y codificadores ------------------
models = {
    "Linear Regression": load('models/regression/linear_regression_model.pkl'),
    "Random Forest": load('models/regression/random_forest_model.pkl')
}

encoder_clima = load('encoders/regression/clima_encoder.pkl')  # OneHotEncoder
custom_order_dia = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']

# ------------------ Inputs desde el sidebar ------------------
st.sidebar.title("🔧 Configuración")
selected_model_name = st.sidebar.selectbox("Modelo de regresión", list(models.keys()))
dia = st.sidebar.selectbox("Día", custom_order_dia)
hora = st.sidebar.slider("Hora (formato 24h)", min_value=8, max_value=21, value=14)
clima = st.sidebar.selectbox("Clima", encoder_clima.categories_[0])

# ------------------ Título principal ------------------
st.title("📈 Predicción de Espacios Disponibles")

# ------------------ Preprocesamiento de entrada ------------------
dia_encoded = custom_order_dia.index(dia)
clima_encoded = encoder_clima.transform(pd.DataFrame([[clima]], columns=['clima']))

X_input = pd.DataFrame([[dia_encoded, hora]], columns=['Día', 'Hora'])
X_full = pd.concat(
    [X_input.reset_index(drop=True), pd.DataFrame(clima_encoded, columns=encoder_clima.get_feature_names_out(['clima']))],
    axis=1
)

# ------------------ Predicción ------------------
model = models[selected_model_name]
prediction = model.predict(X_full)[0]
espacios_disponibles = max(0, min(round(prediction), 20))

# ------------------ Mostrar resultado ------------------
summary_df = pd.DataFrame({
    "Modelo": [selected_model_name],
    "Día": [dia],
    "Hora": [f"{hora}:00"],
    "Clima": [clima]
})

st.markdown("### Resumen de la predicción")
st.dataframe(summary_df, use_container_width=True)
st.success(f" Espacios disponibles: **{espacios_disponibles}** de 20")

# ------------------ Gráficas adicionales ------------------
st.markdown("---")
st.markdown("## 📊 Análisis de desempeño del modelo")

# ⚠️ Generamos datos simulados para visualización
X_sample = pd.DataFrame({
    'Día': np.random.randint(0, 5, 100),
    'Hora': np.random.randint(8, 22, 100)
})
clima_sample = np.random.choice(encoder_clima.categories_[0], 100)
clima_encoded_sample = encoder_clima.transform(pd.DataFrame(clima_sample, columns=['clima']))
X_sample_encoded = pd.concat(
    [X_sample.reset_index(drop=True),
     pd.DataFrame(clima_encoded_sample, columns=encoder_clima.get_feature_names_out(['clima']))],
    axis=1
)

y_pred = model.predict(X_sample_encoded)
y_true = y_pred + np.random.normal(0, 2, 100)  # Simulamos valores reales con ruido
errors = y_true - y_pred

# ------------------ Gráfico de errores residuales ------------------
st.markdown("### 🔻 Gráfica de errores residuales")
fig_error, ax_error = plt.subplots(figsize=(8, 5))
sns.histplot(errors, kde=True, bins=20, ax=ax_error, color='#FF6F61')
ax_error.axvline(0, color='blue', linestyle='--', label='Sin error')
ax_error.set_title("Distribución de errores (residuales)", fontsize=14)
ax_error.set_xlabel("Error = Real - Predicho")
ax_error.legend()
st.pyplot(fig_error)

# ------------------ Gráfico de valores reales vs predichos ------------------
st.markdown("### 🔹 Valores reales vs. predichos")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=y_true, y=y_pred, ax=ax_scatter, color='#3B8EEA', s=40)
ax_scatter.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Línea ideal')
ax_scatter.set_xlabel("Valor real", fontsize=12)
ax_scatter.set_ylabel("Valor predicho", fontsize=12)
ax_scatter.set_title("Comparación entre valores reales y predichos", fontsize=14)
ax_scatter.legend()
st.pyplot(fig_scatter)
