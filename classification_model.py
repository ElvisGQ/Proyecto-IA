import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from catboost import CatBoostClassifier
import os

# Create directories if they don't exist
os.makedirs('models/classification', exist_ok=True)
os.makedirs('encoders/classification', exist_ok=True)
os.makedirs('data/classification', exist_ok=True)  # For test data

# Load CSV from local path
df = pd.read_csv('parking.csv')

# Drop unneeded column
df = df.drop(columns=['Fecha'])

# Encode categorical variables
custom_order_dia = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
le_dia = LabelEncoder()
le_dia.fit(custom_order_dia)
df['Día'] = le_dia.transform(df['Día'])

le_clima = LabelEncoder()
le_estado = LabelEncoder()
df['Clima'] = le_clima.fit_transform(df['Clima'])
df['Estado'] = le_estado.fit_transform(df['Estado'])

# Feature matrix and labels
X = df[['Día', 'Hora', 'Espacio', 'Clima']]
y = df['Estado']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ SAVE TEST DATA FOR STREAMLIT APP
test_data = X_test.copy()
test_data['Estado'] = y_test  # Add true labels
test_data.to_csv('data/classification/test_data.csv', index=False)
print(f"✅ Test data saved: {len(test_data)} samples in 'data/test_data.csv'")

# ✅ Polynomial features for Logistic Regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Initialize and train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Initialize and train RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Initialize and train Logistic Regression model with polynomial features
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_poly, y_train)

# ✅ Initialize and train CatBoost model
cat_model = CatBoostClassifier(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)

# Predictions for XGBoost
xgb_y_pred = xgb_model.predict(X_test)

# Predictions for RandomForest
rf_y_pred = rf_model.predict(X_test)

# ✅ Predictions for Logistic Regression (with polynomial features)
lr_y_pred = lr_model.predict(X_test_poly)

# ✅ Predictions for CatBoost
cat_y_pred = cat_model.predict(X_test)

# Evaluation for XGBoost
print("XGBoost - Accuracy:", accuracy_score(y_test, xgb_y_pred))
print("XGBoost - Classification Report:")
print(classification_report(y_test, xgb_y_pred))

# Evaluation for RandomForest
print("RandomForest - Accuracy:", accuracy_score(y_test, rf_y_pred))
print("RandomForest - Classification Report:")
print(classification_report(y_test, rf_y_pred))

# ✅ Evaluation for Logistic Regression
print("Logistic Regression - Accuracy:", accuracy_score(y_test, lr_y_pred))
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, lr_y_pred))

# ✅ Evaluation for CatBoost
print("CatBoost - Accuracy:", accuracy_score(y_test, cat_y_pred))
print("CatBoost - Classification Report:")
print(classification_report(y_test, cat_y_pred))

# Example prediction input
sample = pd.DataFrame([{
    'Día': le_dia.transform(['Martes'])[0],
    'Hora': 13,
    'Espacio': 20,
    'Clima': le_clima.transform(['Despejado'])[0]
}])

# Prediction using XGBoost
xgb_prediction = xgb_model.predict(sample)
xgb_estado_predicho = le_estado.inverse_transform(xgb_prediction)
print("\nXGBoost - ¿Espacio disponible?", xgb_estado_predicho[0])

# Prediction using RandomForest
rf_prediction = rf_model.predict(sample)
rf_estado_predicho = le_estado.inverse_transform(rf_prediction)
print("RandomForest - ¿Espacio disponible?", rf_estado_predicho[0])

# ✅ Full prediction pipeline for Logistic Regression
sample_scaled = scaler.transform(sample)       # Apply scaling
sample_poly = poly.transform(sample_scaled)    # Then polynomial features
lr_prediction = lr_model.predict(sample_poly)
lr_estado_predicho = le_estado.inverse_transform(lr_prediction)
print("Logistic Regression - ¿Espacio disponible?", lr_estado_predicho[0])

# ✅ Prediction using CatBoost
cat_prediction = cat_model.predict(sample)
cat_estado_predicho = le_estado.inverse_transform(cat_prediction)
print("CatBoost - ¿Espacio disponible?", cat_estado_predicho[0])

# Save the models and encoders
joblib.dump(xgb_model, 'models/classification/xgboost_parking_model.pkl')
joblib.dump(rf_model, 'models/classification/randomforest_parking_model.pkl')
joblib.dump(lr_model, 'models/classification/logistic_parking_model.pkl')
joblib.dump(cat_model, 'models/classification/catboost_parking_model.pkl')

joblib.dump(le_dia, 'encoders/classification/encoder_dia.pkl')
joblib.dump(le_clima, 'encoders/classification/encoder_clima.pkl')
joblib.dump(le_estado, 'encoders/classification/encoder_estado.pkl')
joblib.dump(poly, 'encoders/classification/poly_transformer.pkl')
joblib.dump(scaler, 'encoders/classification/scaler.pkl')

print("Modelos XGBoost, RandomForest, LogisticRegression, CatBoost y codificadores guardados como archivos .pkl.")

# ✅ OPTIONAL: Save additional evaluation data for detailed analysis
evaluation_results = {
    'XGBoost': {
        'y_true': y_test.tolist(),
        'y_pred': xgb_y_pred.tolist(),
        'accuracy': accuracy_score(y_test, xgb_y_pred)
    },
    'RandomForest': {
        'y_true': y_test.tolist(),
        'y_pred': rf_y_pred.tolist(),
        'accuracy': accuracy_score(y_test, rf_y_pred)
    },
    'LogisticRegression': {
        'y_true': y_test.tolist(),
        'y_pred': lr_y_pred.tolist(),
        'accuracy': accuracy_score(y_test, lr_y_pred)
    },
    'CatBoost': {
        'y_true': y_test.tolist(),
        'y_pred': cat_y_pred.tolist(),
        'accuracy': accuracy_score(y_test, cat_y_pred)
    }
}

import json
with open('data/classification/model_evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("✅ Evaluation results saved to 'data/model_evaluation_results.json'")
print(f"✅ Test data contains {len(test_data)} samples with columns: {list(test_data.columns)}")