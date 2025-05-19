# CO₂-Emissions-Vorhersage für Fahrzeuge

Dieses Projekt wurde im Rahmen des Moduls **End-to-End Machine Learning** umgesetzt. Ziel war es, ein vollständiges Machine-Learning-System zu entwickeln – von der Datenaufbereitung bis zum Deployment eines praxistauglichen Vorhersagemodells.

## Projektziel

Die Anwendung sagt den CO₂-Ausstoss eines Fahrzeugs voraus – basierend auf technischen Angaben wie Motorgrösse, Zylinderanzahl und Getriebeart.

---

## Verwendete Datenquellen

Zwei öffentlich zugängliche Datensätze wurden kombiniert:

1. **CO₂ Emissions in Kanada**  
   Quelle: [Kaggle – Canadian Vehicle Emissions](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)

2. **Fuel Economy Daten USA (EPA)**  
   Quelle: [fueleconomy.gov](https://www.fueleconomy.gov/feg/download.shtml)

Die Datensätze wurden anhand der Merkmale **Hersteller (Make)** und **Modell (Model)** zusammengeführt.

---

## ⚙Feature Engineering

- Vereinheitlichung von Schreibweisen (Grossbuchstaben, Leerzeichen entfernen)
- Auswahl technischer Merkmale:
  - `Engine_Size_L`
  - `Cylinders`
  - `Transmission`
- One-Hot-Encoding für Getriebearten

---

## 🧠 Modelle

Zwei Modelle wurden trainiert und evaluiert:

| Modell            | RMSE  | R²     |
|-------------------|-------|--------|
| Linear Regression | 23.36 | 0.836  |
| Random Forest     | 18.57 | 0.869  |

**Random Forest** erzielte die besseren Werte in Bezug auf Fehler (RMSE) und erklärte Varianz (R²), daher wurde es für das Deployment ausgewählt.

---

##App
[Demo](https://huggingface.co/spaces/martivic/co2-predictor)
  

---

## 📁 Projektstruktur
Projekt_End_to_End
├── app.py 
├── co2_rf_model.pkl 
├── merged_vehicle_data.csv 
├── requirements.txt 
├── random_forest.py
├── linear_regression.py
└── README.md 

