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

# Load test data for evaluation (you'll need to save this during training)
# If you don't have test data saved, we'll create a more realistic simulation
try:
    test_data = pd.read_csv('data/classification/test_data.csv')  # Your actual test data
    has_real_data = True
except FileNotFoundError:
    has_real_data = False
    st.warning("⚠️ Test data not found. Using model predictions for demonstration.")

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
prediction_confidence = []

for espacio in range(1, 21):
    input_df = pd.DataFrame([{
        'Día': dia_encoded,
        'Hora': hora,
        'Espacio': espacio,
        'Clima': clima_encoded
    }])

    if selected_model_name == "Logistic Regression":
        input_scaled = scaler.transform(input_df)
        input_df_processed = poly.transform(input_scaled)
    else:
        input_df_processed = input_df

    pred = model.predict(input_df_processed)
    estado = le_estado.inverse_transform(pred)[0]
    predictions.append(estado)

    # Get probabilities
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df_processed)[0]
        probs.append(prob)
        # Store confidence (max probability)
        prediction_confidence.append(max(prob))
    else:
        probs.append([0.5, 0.5])
        prediction_confidence.append(0.5)

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
st.success(f"🚗 Espacios disponibles: **{disponibles}** de 20")

# === 📊 Análisis del Modelo ===
st.markdown("## 📊 Análisis del Modelo Actual")

# Show prediction confidence distribution
st.markdown("### 🎯 Confianza de las Predicciones")
fig_conf, ax_conf = plt.subplots(figsize=(10, 6))
confidence_df = pd.DataFrame({
    'Espacio': [f'P{i+1}' for i in range(20)],
    'Confianza': prediction_confidence,
    'Estado': predictions
})

colors = ['#4CAF50' if estado.lower() == 'disponible' else '#F44336' for estado in predictions]
bars = ax_conf.bar(confidence_df['Espacio'], confidence_df['Confianza'], color=colors, alpha=0.7)
ax_conf.set_xlabel('Espacios de Estacionamiento')
ax_conf.set_ylabel('Confianza del Modelo')
ax_conf.set_title('Confianza de Predicción por Espacio')
ax_conf.set_ylim(0, 1)
plt.xticks(rotation=45)

# Add confidence threshold line
ax_conf.axhline(y=0.7, color='orange', linestyle='--', alpha=0.8, label='Umbral recomendado (70%)')
ax_conf.legend()

# Add value labels on bars
for bar, conf in zip(bars, prediction_confidence):
    height = bar.get_height()
    ax_conf.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
st.pyplot(fig_conf)

# Show current predictions summary
st.markdown("### 📈 Resumen de Predicciones Actuales")
pred_summary = pd.DataFrame(predictions, columns=['Estado']).value_counts().reset_index()
pred_summary.columns = ['Estado', 'Cantidad']

fig_summary, ax_summary = plt.subplots(figsize=(8, 6))
colors_pie = ['#4CAF50' if estado.lower() == 'disponible' else '#F44336' 
              for estado in pred_summary['Estado']]
wedges, texts, autotexts = ax_summary.pie(pred_summary['Cantidad'], 
                                         labels=pred_summary['Estado'],
                                         colors=colors_pie,
                                         autopct='%1.1f%%',
                                         startangle=90)
ax_summary.set_title(f'Distribución de Estados - {dia} a las {hora}:00 hrs')
st.pyplot(fig_summary)

# === Model Evaluation Section ===
if has_real_data:
    st.markdown("## 🔍 Evaluación del Modelo con Datos Reales")
    
    # Filter test data for current conditions or use all test data
    current_test = test_data[
        (test_data['Día'] == dia_encoded) & 
        (test_data['Clima'] == clima_encoded)
    ] if len(test_data) > 100 else test_data.sample(min(100, len(test_data)))
    
    if len(current_test) > 0:
        # Prepare test data
        X_test = current_test[['Día', 'Hora', 'Espacio', 'Clima']]
        y_true = current_test['Estado']  # Assuming 'Estado' is the target column
        
        # Make predictions on test data
        if selected_model_name == "Logistic Regression":
            X_test_scaled = scaler.transform(X_test)
            X_test_processed = poly.transform(X_test_scaled)
        else:
            X_test_processed = X_test
            
        y_pred = model.predict(X_test_processed)
        y_pred_labels = le_estado.inverse_transform(y_pred)
        
        # Get true labels
        if isinstance(y_true.iloc[0], (int, np.integer)):
            y_true_labels = le_estado.inverse_transform(y_true)
        else:
            y_true_labels = y_true.values
        
        # Confusion Matrix
        st.markdown("### 📊 Matriz de Confusión")
        labels = le_estado.classes_
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax_cm)
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        ax_cm.set_title(f'Matriz de Confusión - {selected_model_name}')
        st.pyplot(fig_cm)
        
        # Classification metrics
        st.markdown("### 📈 Métricas de Clasificación")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_labels, y_pred_labels, labels=labels, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Clase': labels,
            'Precisión': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Soporte': support
        })
        
        st.dataframe(metrics_df.round(3))
        
        # Metrics visualization
        fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
        bar_width = 0.25
        x = np.arange(len(labels))
        
        bars1 = ax_metrics.bar(x - bar_width, precision, bar_width, label='Precisión', alpha=0.8)
        bars2 = ax_metrics.bar(x, recall, bar_width, label='Recall', alpha=0.8)
        bars3 = ax_metrics.bar(x + bar_width, f1, bar_width, label='F1-Score', alpha=0.8)
        
        ax_metrics.set_xlabel('Clases')
        ax_metrics.set_ylabel('Puntuación')
        ax_metrics.set_title('Métricas de Rendimiento por Clase')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(labels)
        ax_metrics.set_ylim(0, 1.1)
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig_metrics)
        
        # ROC Curve (if model has predict_proba)
        if hasattr(model, "predict_proba"):
            st.markdown("### 📉 Curva ROC")
            
            # Get probabilities for test data
            y_proba = model.predict_proba(X_test_processed)
            
            # Convert to binary classification for ROC (Disponible vs Ocupado)
            y_true_bin = [1 if label.lower() == 'disponible' else 0 for label in y_true_labels]
            disponible_idx = list(le_estado.classes_).index('Disponible') if 'Disponible' in le_estado.classes_ else 0
            y_scores = y_proba[:, disponible_idx]
            
            fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'{selected_model_name} (AUC = {roc_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Clasificador Aleatorio')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('Tasa de Falsos Positivos')
            ax_roc.set_ylabel('Tasa de Verdaderos Positivos')
            ax_roc.set_title('Curva ROC - Clasificación Disponible/Ocupado')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
            
            # Show AUC interpretation
            if roc_auc >= 0.9:
                st.success(f"🎯 Excelente rendimiento: AUC = {roc_auc:.3f}")
            elif roc_auc >= 0.8:
                st.info(f"👍 Buen rendimiento: AUC = {roc_auc:.3f}")
            elif roc_auc >= 0.7:
                st.warning(f"⚠️ Rendimiento moderado: AUC = {roc_auc:.3f}")
            else:
                st.error(f"❌ Rendimiento bajo: AUC = {roc_auc:.3f}")

else:
    st.markdown("## 💡 Para Evaluación Completa")
    st.info("""
    **Para mostrar métricas de evaluación reales:**
    
    1. Guarda tus datos de prueba durante el entrenamiento:
    ```python
    # Durante el entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Después del entrenamiento, guarda los datos de prueba
    test_data = X_test.copy()
    test_data['Estado'] = y_test
    test_data.to_csv('data/test_data.csv', index=False)
    ```
    
    2. O carga un dataset separado de validación con las columnas:
       - Día, Hora, Espacio, Clima, Estado
    """)

# === 📖 Interpretación ===
st.markdown("---")
st.markdown("## 📖 Interpretación de Resultados")

with st.expander("🔍 Ver explicación detallada"):
    st.markdown("""
    ### 🎯 Confianza de Predicciones
    - **Verde**: Espacios predichos como disponibles
    - **Rojo**: Espacios predichos como ocupados  
    - **Altura de barras**: Nivel de confianza del modelo (0-1)
    - **Línea naranja**: Umbral recomendado de confianza (70%)
    
    ### 📊 Matriz de Confusión
    - **Diagonal principal**: Predicciones correctas
    - **Fuera de diagonal**: Errores de clasificación
    - **Valores altos en diagonal**: Mejor rendimiento
    
    ### 📈 Métricas de Clasificación
    - **Precisión**: De todas las predicciones positivas, ¿cuántas fueron correctas?
    - **Recall**: De todos los casos positivos reales, ¿cuántos detectó el modelo?
    - **F1-Score**: Promedio armónico de precisión y recall
    
    ### 📉 Curva ROC y AUC
    - **AUC cercano a 1**: Excelente capacidad de discriminación
    - **AUC = 0.5**: Rendimiento aleatorio
    - **Curva hacia esquina superior izquierda**: Mejor rendimiento
    """)

# Model comparison section
st.markdown("---")
st.markdown("## ⚖️ Comparación de Modelos")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Modelo Actual")
    st.info(f"**{selected_model_name}**")
    st.write(f"Espacios disponibles: {disponibles}/20")
    avg_confidence = np.mean(prediction_confidence)
    st.write(f"Confianza promedio: {avg_confidence:.2%}")

with col2:
    st.markdown("### Recomendaciones")
    if avg_confidence < 0.7:
        st.warning("🔄 Considera reentrenar el modelo con más datos")
    else:
        st.success("✅ El modelo muestra buena confianza")
    
    st.write("**Consejos para mejorar:**")
    st.write("• Aumentar datos de entrenamiento")
    st.write("• Incluir más características (día festivo, eventos)")
    st.write("• Validar con datos más recientes")