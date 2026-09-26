import streamlit as st
import pandas as pd
import mlflow.sklearn

# 1. Titel & Beschreibung der Web-App
st.title("🧠 Personality Type Predictor")
st.write("""
Diese Anwendung prognostiziert deinen Persönlichkeitstyp basierend auf dem Big-Five-(OCEAN)-Test.
Bitte beantworte die folgenden Fragen und gib deine demografischen Daten an.
""")

# 2. RUN_ID deines Champions hier einfügen!
# Ersetze den Platzhalter durch deine tatsächliche Run ID aus MLflow!
RUN_ID = "7544cecc331849b0ba5adc59ab03a06b"

# Modell aus MLflow laden (mit Caching für schnelle Ladezeiten)
@st.cache_resource
def load_champion_model(run_id):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

try:
    model = load_champion_model(RUN_ID)
    st.success("Champion-Modell erfolgreich aus MLflow geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden des Modells. Bitte prüfe deine RUN_ID! Fehler: {e}")

# 3. Formular für Benutzereingaben
st.header("1. Demografische Angaben")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Alter", min_value=10, max_value=100, value=25)
with col2:
    gender = st.selectbox("Geschlecht", ["Male", "Female", "Other"])
with col3:
    hand = st.selectbox("Schreibhand", ["Right", "Left", "Both"])

st.header("2. Fragebogen (1 = Stimme nicht zu ... 5 = Stimme zu)")

# Wörterbuch mit den 19 Fragen aus dem Codebook
questions = {
    "N1": "Ich gerate leicht in Stress.",
    "N2": "Ich bin die meiste Zeit entspannt.",
    "N3": "Ich mache mir Sorgen um Dinge.",
    "N4": "Ich fühle mich selten niedergeschlagen.",
    "N5": "Ich lasse mich leicht aus der Ruhe bringen.",
    "N6": "Ich werde leicht wütend/aufgeregt.",
    "N7": "Meine Stimmung ändert sich oft.",
    "N8": "Ich habe häufige Stimmungsschwankungen.",
    "N9": "Ich werde leicht irritiert.",
    "N10": "Ich fühle mich oft niedergeschlagen.",
    "E1": "Ich bin der Mittelpunkt der Party.",
    "E3": "Ich fühle mich wohl unter Menschen.",
    "E4": "Ich halte mich eher im Hintergrund.",
    "E5": "Ich starte Gespräche von mir aus.",
    "E7": "Ich spreche auf Partys mit vielen verschiedenen Leuten.",
    "E9": "Es macht mir nichts aus, im Mittelpunkt der Aufmerksamkeit zu stehen.",
    "E10": "Ich bin leise/ruhig gegenüber Fremden.",
    "C4": "Ich bringe Dinge durcheinander / mache Unordnung.",
    "A4": "Ich fühle mit den Gefühlen anderer mit."
}

user_answers = {}
for code, question_text in questions.items():
    user_answers[code] = st.slider(f"{code}: {question_text}", 1, 5, 3)

# 4. Vorhersage ausführen
if st.button("Persönlichkeitstyp vorhersagen", type="primary"):
    # Eingaben in ein DataFrame mit exakt einer Zeile verpacken
    input_data = user_answers.copy()
    input_data["age"] = age
    input_data["gender"] = gender
    input_data["hand"] = hand
    
    input_df = pd.DataFrame([input_data])
    
    # Vorhersage mit der geladenen Pipeline
    prediction = model.predict(input_df)[0]
    
    st.subheader(f"Ergebnis: Dein vorhergesagter Typ ist **{prediction}**")