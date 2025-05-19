import pandas as pd

# xls-Datei einlesen
epa_df = pd.read_excel("vehicles.xlsx")

# optional: als CSV speichern für spätere einfache Nutzung
epa_df.to_csv("epa_cleaned.csv", index=False)
