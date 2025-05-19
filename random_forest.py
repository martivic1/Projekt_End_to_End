# 📦 Bibliotheken importieren
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# 📂 Datensatz laden
df = pd.read_csv("merged_vehicle_data.csv")

# 🧹 Spalten umbenennen
df = df.rename(columns={
    "Engine_Size_L_x": "Engine_Size_L",
    "Cylinders_x": "Cylinders",
    "Transmission_x": "Transmission"
})

# 🎯 Ziel und Features festlegen
features = ["Engine_Size_L", "Cylinders", "Transmission"]
target = "CO2_Canada"

# 🔀 Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# ⚙️ Preprocessing Pipeline
numeric_features = ["Engine_Size_L", "Cylinders"]
categorical_features = ["Transmission"]

preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 🧠 Modell definieren
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# 📈 Modell trainieren
rf_model.fit(X_train, y_train)

# 📦 Modell speichern
joblib.dump(rf_model, "co2_rf_model.pkl")

# 🔍 Vorhersagen machen
y_pred_rf = rf_model.predict(X_test)

# 📊 Modell bewerten
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²:   {r2_rf:.3f}")
