import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models/regression', exist_ok=True)
os.makedirs('encoders/regression', exist_ok=True)
os.makedirs('data/regression', exist_ok=True)

# Load and process data
df = pd.read_csv('parking.csv')

# Transform data structure
df_available = df[df['Estado'] == 'Disponible']
spaces_available = df_available.groupby(['Día', 'Hora'])['Espacio'].count().reset_index(name='espacios_disponibles')
clima_mode = df_available.groupby(['Día', 'Hora'])['Clima'].agg(lambda x: x.mode()[0]).reset_index(name='clima')
df_result = pd.merge(spaces_available, clima_mode, on=['Día', 'Hora'])

# Order days
days_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
df_result['Día'] = pd.Categorical(df_result['Día'], categories=days_order, ordered=True)
df_result = df_result.sort_values(by=['Día', 'Hora'])

# Feature engineering
df_result['Día_encoded'] = df_result['Día'].astype('category').cat.codes
df_result['Hora'] = pd.to_numeric(df_result['Hora'], errors='coerce')

# One-hot encode clima
encoder = OneHotEncoder(categories=[['Despejado', 'Lluvioso', 'Nublado']], drop='first', sparse_output=False, handle_unknown='ignore')
clima_encoded = encoder.fit_transform(df_result[['clima']])

# Create feature matrix with proper column names
clima_columns = encoder.get_feature_names_out(['clima'])
clima_encoded_df = pd.DataFrame(clima_encoded, columns=clima_columns, index=df_result.index)

X = pd.concat([
    df_result[['Día_encoded', 'Hora']].rename(columns={'Día_encoded': 'Día'}),
    clima_encoded_df
], axis=1)

y = df_result['espacios_disponibles']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save test data
regression_test_data = X_test.copy()
regression_test_data['espacios_disponibles'] = y_test
regression_test_data.to_csv('data/regression/test_data.csv', index=False)

# Save metadata
regression_metadata = {
    'feature_columns': list(X.columns),
    'target_column': 'espacios_disponibles',
    'days_order': days_order,
    'clima_categories': list(encoder.categories_[0]),
    'encoded_clima_features': list(encoder.get_feature_names_out(['clima'])),
    'total_spaces': 20,
    'data_shape': {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X.columns)
    }
}

with open('data/regression/metadata.json', 'w') as f:
    json.dump(regression_metadata, f, indent=2)

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Cross-validation
n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(rf_model, X, y, cv=kfold, scoring='neg_mean_squared_error')

mse_scores_lr = -cv_scores_lr
mse_scores_rf = -cv_scores_rf

# Save evaluation results
regression_evaluation = {
    'LinearRegression': {
        'y_true': y_test.tolist(),
        'y_pred': y_pred_lr.tolist(),
        'mse': mse_lr,
        'mae': mae_lr,
        'r2': r2_lr,
        'cv_mse_mean': np.mean(mse_scores_lr),
        'cv_mse_std': np.std(mse_scores_lr)
    },
    'RandomForest': {
        'y_true': y_test.tolist(),
        'y_pred': y_pred_rf.tolist(),
        'mse': mse_rf,
        'mae': mae_rf,
        'r2': r2_rf,
        'cv_mse_mean': np.mean(mse_scores_rf),
        'cv_mse_std': np.std(mse_scores_rf)
    }
}

with open('data/regression/evaluation_results.json', 'w') as f:
    json.dump(regression_evaluation, f, indent=2)

# Save models and encoder
joblib.dump(lr_model, 'models/regression/linear_regression_model.pkl')
joblib.dump(rf_model, 'models/regression/random_forest_model.pkl')
joblib.dump(encoder, 'encoders/regression/clima_encoder.pkl')

# Generate sample predictions using DataFrame to avoid feature name warning
sample_predictions = []
for day_idx, day_name in enumerate(days_order):
    for hour in range(8, 22):
        for clima_cat in encoder.categories_[0]:
            # Create sample input as DataFrame with proper column names
            sample_input = pd.DataFrame([[day_idx, hour]], columns=['Día', 'Hora'])
            
            # Add clima encoding
            clima_encoded_sample = encoder.transform([[clima_cat]])
            clima_df = pd.DataFrame(clima_encoded_sample, columns=clima_columns)
            
            # Combine features
            sample_input_full = pd.concat([sample_input, clima_df], axis=1)
            
            # Make predictions
            lr_pred = lr_model.predict(sample_input_full)[0]
            rf_pred = rf_model.predict(sample_input_full)[0]
            
            sample_predictions.append({
                'Día': day_name,
                'Hora': hour,
                'Clima': clima_cat,
                'LR_Prediction': max(0, min(20, round(lr_pred))),
                'RF_Prediction': max(0, min(20, round(rf_pred)))
            })

sample_predictions_df = pd.DataFrame(sample_predictions)
sample_predictions_df.to_csv('data/regression/sample_predictions.csv', index=False)

# Results
print(f"Linear Regression - MSE: {mse_lr:.2f}, MAE: {mae_lr:.2f}, R²: {r2_lr:.3f}")
print(f"Random Forest - MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.3f}")
print(f"CV Results - LR MSE: {np.mean(mse_scores_lr):.2f}±{np.std(mse_scores_lr):.2f}")
print(f"CV Results - RF MSE: {np.mean(mse_scores_rf):.2f}±{np.std(mse_scores_rf):.2f}")
print(f"Best model: {'Random Forest' if r2_rf > r2_lr else 'Linear Regression'}")
print("✅ All files saved successfully")