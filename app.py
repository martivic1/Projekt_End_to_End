import gradio as gr
import pandas as pd
import joblib


model = joblib.load("co2_rf_model.pkl")

def predict_co2(engine_size, cylinders, transmission):
    df = pd.DataFrame([[engine_size, cylinders, transmission]],
                      columns=["Engine_Size_L", "Cylinders", "Transmission"])
    prediction = model.predict(df)[0]
    return f"Geschätzte CO₂-Emissionen: {prediction:.1f} g/km"

transmission_options = [
    "Auto(S6)",
    "Auto(S8)",
    "Auto(AV)",
    "Manual(M6)",
    "Auto(AM-S7)",
    "Auto(A8)"
]

interface = gr.Interface(
    fn=predict_co2,
    inputs=[
        gr.Number(label="Motorgrösse (L)", minimum=1),
        gr.Number(label="Zylinderanzahl", minimum=1, step=1),
        gr.Dropdown(choices=transmission_options, label="Getriebeart")
    ],
    outputs="text",
    title="CO₂-Emissions-Vorhersage",
    description="Diese App sagt den CO₂-Ausstoss eines Fahrzeugs voraus. Wähle technische Daten und Getriebeart."
)

if __name__ == "__main__":
    interface.launch()
