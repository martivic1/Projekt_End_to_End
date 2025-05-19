import pandas as pd

# Dateien laden
epa_df = pd.read_csv("epa_cleaned.csv")
canada_df = pd.read_csv("CO2 Emissions_Canada.csv")

# Relevante Spalten extrahieren
epa_small = epa_df[["Model Year", "Mfr Name", "Carline", "Eng Displ", "# Cyl", "Transmission",
                    "Comb CO2 Rounded Adjusted (as shown on FE Label)"]].copy()
canada_small = canada_df[["Make", "Model", "Engine Size(L)", "Cylinders", "Transmission",
                          "CO2 Emissions(g/km)"]].copy()

# Umbenennen für Klarheit
epa_small.columns = ["Year", "Make", "Model", "Engine_Size_L", "Cylinders", "Transmission", "CO2_US"]
canada_small.columns = ["Make", "Model", "Engine_Size_L", "Cylinders", "Transmission", "CO2_Canada"]

# Vereinheitlichen (Grossbuchstaben, keine Leerzeichen)
epa_small["Make"] = epa_small["Make"].str.upper().str.strip()
epa_small["Model"] = epa_small["Model"].str.upper().str.strip()
canada_small["Make"] = canada_small["Make"].str.upper().str.strip()
canada_small["Model"] = canada_small["Model"].str.upper().str.strip()

# Merge durchführen
merged_df = pd.merge(canada_small, epa_small, on=["Make", "Model"], how="inner")

# Ergebnis anzeigen
print(merged_df.head())
print(f"Zusammengeführte Zeilen: {len(merged_df)}")

# Zusammengeführte Datei speichern
merged_df.to_csv("merged_vehicle_data.csv", index=False)


