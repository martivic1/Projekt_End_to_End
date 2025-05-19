# 📦 Bibliotheken importieren
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 📈 Modell trainieren
linear_model.fit(X_train, y_train)

# 🔍 Vorhersagen machen
y_pred_lr = linear_model.predict(X_test)

# 📊 Modell bewerten
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression:")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R²:   {r2_lr:.3f}")
