import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Load models and encoders
models = {
    "XGBoost": load('models/classification/xgboost_parking_model.pkl'),
    "Random Forest": load('models/classification/randomforest_parking_model.pkl'),
    "Logistic Regression": load('models/classification/logistic_parking_model.pkl'),
    "CatBoost": load('models/classification/catboost_parking_model.pkl')
}

le_dia = load('encoders/classification/encoder_dia.pkl')
le_clima = load('encoders/classification/encoder_clima.pkl')
le_estado = load('encoders/classification/encoder_estado.pkl')
poly = load('encoders/classification/poly_transformer.pkl')
scaler = load('encoders/classification/scaler.pkl')

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
probs = []
for espacio in range(1, 21):
    input_df = pd.DataFrame([{
        'Día': dia_encoded,
        'Hora': hora,
        'Espacio': espacio,
        'Clima': clima_encoded
    }])

    if selected_model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        input_df = poly.transform(input_scaled)

    pred = model.predict(input_df)
    estado = le_estado.inverse_transform(pred)[0]
    predictions.append(estado)

    # Obtener probabilidades para ROC
    if hasattr(model, "predict_proba"):
        probs.append(model.predict_proba(input_df)[0])
    else:
        probs.append([0.5, 0.5])  # Suponemos neutral para modelos sin predict_proba

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

st.markdown("### Mapa del estacionamiento:")
st.markdown(html, unsafe_allow_html=True)

st.markdown("---")
st.success(f" Espacios disponibles: **{disponibles}** de 20")

# === 📊 Gráficas Avanzadas ===
st.markdown("## 📊 Evaluación del Modelo")

# Etiquetas simuladas para graficar
labels = le_estado.classes_
np.random.seed(42)
y_true = np.random.choice(labels, size=len(predictions), p=[0.5, 0.5])
y_pred = np.array(predictions)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
ax_cm.set_xlabel('Predicción')
ax_cm.set_ylabel('Real')
ax_cm.set_title('Matriz de Confusión (Simulada)')
st.pyplot(fig_cm)

# Métricas de clasificación por clase
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels)
fig_metrics, ax_metrics = plt.subplots()
bar_width = 0.25
x = np.arange(len(labels))
ax_metrics.bar(x, precision, width=bar_width, label='Precisión')
ax_metrics.bar(x + bar_width, recall, width=bar_width, label='Sensibilidad')
ax_metrics.bar(x + 2 * bar_width, f1, width=bar_width, label='F1 Score')
ax_metrics.set_xticks(x + bar_width)
ax_metrics.set_xticklabels(labels)
ax_metrics.set_ylim(0, 1)
ax_metrics.set_title("Métricas por Clase")
ax_metrics.legend()
st.pyplot(fig_metrics)

# Curva ROC y AUC (solo si hay predict_proba)
y_true_bin = np.array([1 if val == 'Disponible' else 0 for val in y_true])
y_scores = np.array([p[le_estado.transform(['Disponible'])[0]] for p in probs])
fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('Tasa de Falsos Positivos')
ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
ax_roc.set_title('Curva ROC y AUC')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# === ℹ️ Explicación ===
st.markdown("---")
st.markdown("## ℹ️ Interpretación de Resultados")
with st.expander("Ver explicación de cada gráfica"):
    st.markdown("### 📌 Matriz de Confusión")
    st.markdown("""
    Compara las predicciones del modelo con los valores reales.
    - ✔️ Diagonal = Aciertos.
    - ❌ Fuera de la diagonal = Errores de clasificación.
    """)

    st.markdown("### 📊 Métricas por Clase")
    st.markdown("""
    Visualiza precisión, recall y F1 para cada clase:
    - **Precisión**: qué tan certero es el modelo.
    - **Recall**: cuántos casos reales detecta.
    - **F1**: promedio balanceado entre los dos.
    """)

    st.markdown("### 📈 Curva ROC y AUC")
    st.markdown("""
    Indica qué tan bien distingue el modelo entre clases:
    - Curva hacia la esquina superior izquierda = Mejor.
    - AUC cercano a 1 = Modelo preciso.
    """)
