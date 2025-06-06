# pages/Regression.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------ Configuración de estilo y límites de imagen ------------------
# Fix for PIL DecompressionBombError
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove PIL image size limit

# Configure matplotlib for better memory usage
plt.rcParams.update({
    "figure.autolayout": True,
    "figure.max_open_warning": 0,
    "figure.dpi": 80,  # Lower DPI to reduce image size
    "savefig.dpi": 80,
    "figure.figsize": (8, 6)  # Standard figure size
})

sns.set_style("darkgrid")

# Function to safely create and display plots
def safe_pyplot(fig, clear_after=True):
    """Safely display pyplot figure and clear memory"""
    try:
        st.pyplot(fig, clear_figure=True)
        if clear_after:
            plt.close(fig)
    except Exception as e:
        st.error(f"Error displaying plot: {str(e)}")
        plt.close(fig)

# ------------------ Cargar modelos y codificadores ------------------
@st.cache_resource
def load_models_and_encoders():
    models = {
        "Linear Regression": load('models/regression/linear_regression_model.pkl'),
        "Random Forest": load('models/regression/random_forest_model.pkl')
    }
    encoder_clima = load('encoders/regression/clima_encoder.pkl')
    return models, encoder_clima

models, encoder_clima = load_models_and_encoders()
custom_order_dia = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']

# Load metadata
try:
    with open('data/regression/metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']
    clima_categories = metadata['clima_categories']
except FileNotFoundError:
    feature_columns = ['Día', 'Hora'] + list(encoder_clima.get_feature_names_out(['clima']))
    clima_categories = list(encoder_clima.categories_[0])

# Load test data for evaluation
@st.cache_data
def load_test_data():
    try:
        test_data = pd.read_csv('data/regression/test_data.csv')
        return test_data, True
    except FileNotFoundError:
        return None, False

test_data, has_real_data = load_test_data()

# ------------------ Inputs desde el sidebar ------------------
st.sidebar.title("🔧 Configuración")
selected_model_name = st.sidebar.selectbox("Modelo de regresión", list(models.keys()))
dia = st.sidebar.selectbox("Día", custom_order_dia)
hora = st.sidebar.slider("Hora (formato 24h)", min_value=8, max_value=21, value=14)
clima = st.sidebar.selectbox("Clima", clima_categories)

# ------------------ Título principal ------------------
st.title("📈 Predicción de Espacios Disponibles")

# ------------------ Preprocesamiento de entrada ------------------
dia_encoded = custom_order_dia.index(dia)
clima_encoded = encoder_clima.transform(pd.DataFrame([[clima]], columns=['clima']))

X_input = pd.DataFrame([[dia_encoded, hora]], columns=['Día', 'Hora'])
X_full = pd.concat([
    X_input.reset_index(drop=True), 
    pd.DataFrame(clima_encoded, columns=encoder_clima.get_feature_names_out(['clima']))
], axis=1)

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
st.success(f"🚗 Espacios disponibles: **{espacios_disponibles}** de 20")

# ------------------ Análisis de desempeño del modelo ------------------
st.markdown("---")
st.markdown("## 📊 Análisis de Desempeño del Modelo")

if has_real_data:
    st.markdown("### 🔍 Evaluación con Datos Reales")
    

    # Determine target column name
    target_col = None
    possible_targets = ['espacios_disponibles', 'Espacios_Disponibles', 'espacios_disponibles']
    for col in possible_targets:
        if col in test_data.columns:
            target_col = col
            break
    
    if target_col is None:
        st.error("❌ No se encontró la columna objetivo en los datos de prueba")
        st.write("Columnas disponibles:", list(test_data.columns))
        has_real_data = False
    else:
        # Use a sample of test data
        current_test = test_data.sample(min(100, len(test_data)), random_state=42)
        
        # Prepare features for prediction
        X_test = current_test[feature_columns].copy()
        y_true = current_test[target_col].values
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Display metrics
        st.markdown("### 📈 Métricas de Rendimiento")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # Interpretation of R²
        if r2 >= 0.9:
            st.success(f"🎯 Excelente ajuste: R² = {r2:.3f}")
        elif r2 >= 0.7:
            st.info(f"👍 Buen ajuste: R² = {r2:.3f}")
        elif r2 >= 0.5:
            st.warning(f"⚠️ Ajuste moderado: R² = {r2:.3f}")
        else:
            st.error(f"❌ Ajuste bajo: R² = {r2:.3f}")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # ------------------ Gráfico de errores residuales ------------------
        st.markdown("### 🔻 Análisis de Residuales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))  # Smaller figure size
            sns.histplot(residuals, kde=True, bins=20, ax=ax_hist, color='#FF6F61')
            ax_hist.axvline(0, color='blue', linestyle='--', alpha=0.8, label='Residual = 0')
            ax_hist.set_title("Distribución de Residuales", fontsize=12)
            ax_hist.set_xlabel("Residual (Real - Predicho)")
            ax_hist.set_ylabel("Frecuencia")
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            safe_pyplot(fig_hist)
        
        with col2:
            fig_qq, ax_qq = plt.subplots(figsize=(6, 4))  # Smaller figure size
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax_qq)
            ax_qq.set_title("Q-Q Plot de Residuales", fontsize=12)
            ax_qq.grid(True, alpha=0.3)
            safe_pyplot(fig_qq)
        
        # ------------------ Gráfico de valores reales vs predichos ------------------
        st.markdown("### 🔹 Valores Reales vs. Predichos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))  # Smaller figure size
            sns.scatterplot(x=y_true, y=y_pred, ax=ax_scatter, color='#3B8EEA', s=40, alpha=0.7)
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                          color='red', linestyle='--', alpha=0.8, label='Predicción perfecta')
            
            ax_scatter.set_xlabel("Valor Real", fontsize=10)
            ax_scatter.set_ylabel("Valor Predicho", fontsize=10)
            ax_scatter.set_title("Comparación Valores Reales vs Predichos", fontsize=12)
            ax_scatter.legend()
            ax_scatter.grid(True, alpha=0.3)
            safe_pyplot(fig_scatter)
        
        with col2:
            fig_residual_scatter, ax_residual = plt.subplots(figsize=(6, 4))  # Smaller figure size
            sns.scatterplot(x=y_pred, y=residuals, ax=ax_residual, color='#FF6F61', s=40, alpha=0.7)
            ax_residual.axhline(y=0, color='blue', linestyle='--', alpha=0.8)
            ax_residual.set_xlabel("Valores Predichos", fontsize=10)
            ax_residual.set_ylabel("Residuales", fontsize=10)
            ax_residual.set_title("Residuales vs Valores Predichos", fontsize=12)
            ax_residual.grid(True, alpha=0.3)
            safe_pyplot(fig_residual_scatter)
        
        # ------------------ Análisis por variables ------------------
        st.markdown("### 📊 Análisis por Variables")
        
        # Performance by day (if Día column exists)
        if 'Día' in current_test.columns:
            day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
            day_performance = []
            
            for day_idx in range(5):
                day_mask = current_test['Día'] == day_idx
                if day_mask.sum() > 0:
                    day_indices = day_mask[day_mask].index
                    test_indices = [i for i, idx in enumerate(current_test.index) if idx in day_indices]
                    
                    if len(test_indices) > 0:
                        day_y_true = y_true[test_indices]
                        day_y_pred = y_pred[test_indices]
                        
                        day_r2 = r2_score(day_y_true, day_y_pred) if len(day_y_true) > 1 else 0
                        day_mae = mean_absolute_error(day_y_true, day_y_pred)
                        day_performance.append({
                            'Día': day_names[day_idx],
                            'R²': day_r2,
                            'MAE': day_mae,
                            'Muestras': len(test_indices)
                        })
            
            if day_performance:
                performance_df = pd.DataFrame(day_performance)
                performance_df['R²'] = performance_df['R²'].clip(lower=0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_day_r2, ax_day_r2 = plt.subplots(figsize=(6, 4))  # Smaller figure size
                    bars = ax_day_r2.bar(performance_df['Día'], performance_df['R²'], 
                                        color='#3B8EEA', alpha=0.7)
                    ax_day_r2.set_title('R² Score por Día', fontsize=12)
                    ax_day_r2.set_ylabel('R² Score')
                    ax_day_r2.set_ylim(0, 1)
                    ax_day_r2.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, r2_val in zip(bars, performance_df['R²']):
                        height = bar.get_height()
                        ax_day_r2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                      f'{r2_val:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    ax_day_r2.tick_params(axis='x', rotation=45, labelsize=8)
                    safe_pyplot(fig_day_r2)
                
                with col2:
                    fig_day_mae, ax_day_mae = plt.subplots(figsize=(6, 4))  # Smaller figure size
                    bars = ax_day_mae.bar(performance_df['Día'], performance_df['MAE'], 
                                         color='#FF6F61', alpha=0.7)
                    ax_day_mae.set_title('MAE por Día', fontsize=12)
                    ax_day_mae.set_ylabel('Mean Absolute Error')
                    ax_day_mae.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, mae_val in zip(bars, performance_df['MAE']):
                        height = bar.get_height()
                        ax_day_mae.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{mae_val:.2f}', ha='center', va='bottom', fontsize=8)
                    
                    ax_day_mae.tick_params(axis='x', rotation=45, labelsize=8)
                    safe_pyplot(fig_day_mae)
                
                st.markdown("#### Tabla de Rendimiento por Día")
                st.dataframe(performance_df.round(3), use_container_width=True)
        
        # ------------------ Análisis de errores por rango de predicción ------------------
        st.markdown("### 🎯 Análisis de Errores por Rango de Predicción")
        
        # Create bins for predicted values
        pred_bins = pd.cut(y_pred, bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])
        error_by_range = pd.DataFrame({
            'Rango': pred_bins,
            'Error_Abs': np.abs(residuals),
            'Predicción': y_pred,
            'Real': y_true
        })
        
        fig_error_range, ax_error_range = plt.subplots(figsize=(8, 4))  # Smaller figure size
        sns.boxplot(data=error_by_range, x='Rango', y='Error_Abs', ax=ax_error_range)
        ax_error_range.set_title('Distribución de Errores Absolutos por Rango de Predicción', fontsize=12)
        ax_error_range.set_ylabel('Error Absoluto')
        ax_error_range.grid(True, alpha=0.3)
        ax_error_range.tick_params(axis='x', rotation=45, labelsize=8)
        safe_pyplot(fig_error_range)
        
        # Summary statistics
        range_stats = error_by_range.groupby('Rango').agg({
            'Error_Abs': ['mean', 'std', 'count'],
            'Predicción': ['mean', 'min', 'max']
        }).round(3)
        
        st.markdown("#### Estadísticas de Error por Rango")
        st.dataframe(range_stats, use_container_width=True)

if not has_real_data:
    # Fallback to simulated data
    st.markdown("### ⚠️ Usando Datos Simulados")
    
    # Generate realistic simulated data
    np.random.seed(42)
    n_samples = 200
    
    # Create feature matrix matching the expected format
    dias_sim = np.random.randint(0, 5, n_samples)
    horas_sim = np.random.randint(8, 22, n_samples)
    climas_sim = np.random.choice(clima_categories, n_samples)
    
    # Create feature matrix
    X_sim = pd.DataFrame({'Día': dias_sim, 'Hora': horas_sim})
    clima_sim_encoded = encoder_clima.transform(pd.DataFrame(climas_sim, columns=['clima']))
    X_sim_full = pd.concat([
        X_sim.reset_index(drop=True),
        pd.DataFrame(clima_sim_encoded, columns=encoder_clima.get_feature_names_out(['clima']))
    ], axis=1)
    
    # Generate predictions and add realistic noise
    y_pred_sim = model.predict(X_sim_full)
    noise_factor = 0.15 * np.std(y_pred_sim)
    y_true_sim = y_pred_sim + np.random.normal(0, noise_factor, n_samples)
    y_true_sim = np.clip(y_true_sim, 0, 20)
    
    errors = y_true_sim - y_pred_sim
    
    # Calculate metrics
    mse_sim = mean_squared_error(y_true_sim, y_pred_sim)
    rmse_sim = np.sqrt(mse_sim)
    mae_sim = mean_absolute_error(y_true_sim, y_pred_sim)
    r2_sim = r2_score(y_true_sim, y_pred_sim)
    
    # Display simulated metrics
    st.markdown("### 📈 Métricas de Rendimiento (Simuladas)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2_sim:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse_sim:.2f}")
    with col3:
        st.metric("MAE", f"{mae_sim:.2f}")
    with col4:
        st.metric("MSE", f"{mse_sim:.2f}")
    
    # Charts with simulated data
    st.markdown("### 🔻 Distribución de Errores")
    fig_error, ax_error = plt.subplots(figsize=(8, 4))  # Smaller figure size
    sns.histplot(errors, kde=True, bins=20, ax=ax_error, color='#FF6F61')
    ax_error.axvline(0, color='blue', linestyle='--', label='Error = 0')
    ax_error.set_title("Distribución de Errores Residuales", fontsize=12)
    ax_error.set_xlabel("Error (Real - Predicho)")
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)
    safe_pyplot(fig_error)
    
    st.markdown("### 🔹 Valores Reales vs. Predichos")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))  # Smaller figure size
    sns.scatterplot(x=y_true_sim, y=y_pred_sim, ax=ax_scatter, color='#3B8EEA', s=30, alpha=0.7)
    ax_scatter.plot([min(y_true_sim), max(y_true_sim)], [min(y_true_sim), max(y_true_sim)], 
                   color='red', linestyle='--', label='Predicción perfecta')
    ax_scatter.set_xlabel("Valor Real", fontsize=10)
    ax_scatter.set_ylabel("Valor Predicho", fontsize=10)
    ax_scatter.set_title("Comparación entre Valores Reales y Predichos", fontsize=12)
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    safe_pyplot(fig_scatter)

# ------------------ Comparación de modelos ------------------
st.markdown("---")
st.markdown("## ⚖️ Comparación de Modelos")

if len(models) > 1:
    st.markdown("### 🔄 Rendimiento de Todos los Modelos")
    
    model_comparison = []
    for model_name, model_obj in models.items():
        pred_comparison = model_obj.predict(X_full)[0]
        espacios_comparison = max(0, min(round(pred_comparison), 20))
        model_comparison.append({
            'Modelo': model_name,
            'Predicción': espacios_comparison,
            'Predicción_Raw': pred_comparison
        })
    
    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization of model comparison
    fig_comparison, ax_comparison = plt.subplots(figsize=(8, 4))  # Smaller figure size
    bars = ax_comparison.bar(comparison_df['Modelo'], comparison_df['Predicción'], 
                            color=['#3B8EEA', '#FF6F61'][:len(comparison_df)], alpha=0.7)
    ax_comparison.set_title(f'Predicciones por Modelo - {dia} a las {hora}:00', fontsize=12)
    ax_comparison.set_ylabel('Espacios Disponibles Predichos')
    ax_comparison.set_ylim(0, 20)
    ax_comparison.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, pred in zip(bars, comparison_df['Predicción']):
        height = bar.get_height()
        ax_comparison.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                          f'{pred}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    safe_pyplot(fig_comparison)

# ------------------ Interpretación de resultados ------------------
st.markdown("---")
st.markdown("## 📖 Interpretación de Resultados")

with st.expander("🔍 Ver explicación detallada"):
    st.markdown("""
    ### 📊 Métricas de Regresión
    
    - **R² Score**: Indica qué porcentaje de la variabilidad es explicada por el modelo
      - R² > 0.7: Buen ajuste | R² > 0.9: Excelente ajuste
    - **RMSE**: Error cuadrático medio, penaliza más los errores grandes
    - **MAE**: Error absoluto promedio, más robusto a valores atípicos
    
    ### 📈 Análisis de Residuales
    
    - **Histograma**: Debe seguir distribución normal centrada en 0
    - **Q-Q Plot**: Puntos deben seguir línea diagonal
    - **Residuales vs Predichos**: No debe mostrar patrones
    """)

# ------------------ Recomendaciones ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Modelo Actual")
    st.info(f"**{selected_model_name}**")
    st.write(f"Predicción: {espacios_disponibles} espacios")
    
    if has_real_data and 'r2' in locals():
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Error promedio: ±{mae:.1f} espacios")

with col2:
    st.markdown("### 💡 Recomendaciones")
    
    if has_real_data and 'r2' in locals():
        if r2 < 0.5:
            st.warning("🔄 Considera reentrenar el modelo")
        elif mae > 2:
            st.warning("📊 El error promedio es alto")
        else:
            st.success("✅ Buen rendimiento del modelo")
    
    st.write("**Consejos para mejorar:**")
    st.write("• Incluir más variables")
    st.write("• Recopilar más datos históricos")
    st.write("• Considerar efectos estacionales")

# Clear any remaining figures
plt.close('all')