import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Assuming your dataframe is called df
df = pd.read_csv('parking.csv')
pd.concat([df.head(5), df.tail(5)])

# Procesamos los datos del dataset para pasar de esto:

#   Día   |  Hora  | Espacio  |    Estado     |    Clima
#-----------------------------------------------------------
# Lunes   |    8   |    1     |  Disponible    |   Despejado
# Lunes   |    8   |    2     |  Disponible    |   Despejado
# Lunes   |    8   |    3     |  Ocupado       |   Despejado

# A esto:

#   Día   |  Hora  | espacios_disponibles  |    clima
#-----------------------------------------------------------
# Lunes   |    8   |         13            |   Despejado
# Lunes   |    9   |         12            |   Despejado

# Filter available spaces
df_available = df[df['Estado'] == 'Disponible']
spaces_available = df_available.groupby(['Día', 'Hora'])['Espacio'].count().reset_index(name='espacios_disponibles')
clima_mode = df_available.groupby(['Día', 'Hora'])['Clima'].agg(lambda x: x.mode()[0]).reset_index(name='clima')
df_result = pd.merge(spaces_available, clima_mode, on=['Día', 'Hora'])

# Order days of the week
days_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
df_result['Día'] = pd.Categorical(df_result['Día'], categories=days_order, ordered=True)
df_result = df_result.sort_values(by=['Día', 'Hora'])

df_result.head(10)

# Convert categorical to numeric
df_result['Día'] = df_result['Día'].astype('category').cat.codes
df_result['Hora'] = pd.to_numeric(df_result['Hora'], errors='coerce')

# One-hot encode 'clima' without preprocessor
encoder = OneHotEncoder(drop='first', sparse_output=False)
clima_encoded = encoder.fit_transform(df_result[['clima']])

# Get encoded column names
clima_encoded_df = pd.DataFrame(clima_encoded, columns=encoder.get_feature_names_out(['clima']))

# Combine with original features
X = pd.concat([df_result[['Día', 'Hora']].reset_index(drop=True), clima_encoded_df], axis=1)
y = df_result['espacios_disponibles']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Lineal Regression model
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Lineal Regression Test MSE: {mse_lr:.2f} / 20 espacios")
print(f"Lineal Regression R²: {r2_lr}")
print("")
print(f"Random Forest Test MSE: {mse_rf:.2f} / 20 espacios")
print(f"Random Forest R²: {r2_rf}")

# Cross-validation for Linear Regression
n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
mse_scores_lr = -cv_scores_lr
# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
mse_scores_rf = -cv_scores_rf

print(f"\nCross-Validation Results:")
print(f"Linear Regression - Mean MSE: {np.mean(mse_scores_lr):.2f}, Standard Deviation: {np.std(mse_scores_lr):.2f}")
print(f"Random Forest - Mean MSE: {np.mean(mse_scores_rf):.2f}, Standard Deviation: {np.std(mse_scores_rf):.2f}")

# Save
joblib.dump(lr_model, 'models/regression/linear_regression_model.pkl')
joblib.dump(rf_model, 'models/regression/random_forest_model.pkl')
joblib.dump(encoder, 'encoders/regression/clima_encoder.pkl')