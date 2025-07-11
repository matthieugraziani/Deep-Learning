import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model
from utils.preprocess import preprocess_image
from utils.predict import predict_image

CLASS_NAMES = { 0: "Sain", 1: "Tumeur"}

# Charger le modèle
@st.cache_resource
def load_model():
    return keras_load_model("models/model_cnn_seq.keras")

model = load_model()

# Interface Streamlit
st.set_page_config(page_title="Diagnostic Médical IA", layout="centered")
st.title("🩺 Diagnostic d'imagerie médicale assisté par IA - DiagMind.AI")
st.markdown("Envoyez une IRM pour obtenir une prédiction automatique.")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée")

    processed_image = preprocess_image(image)

    try:
        predictions, classe = predict_image(model, processed_image)
        st.write("Prédictions (scores) :", predictions)
        
        classe_nom = CLASS_NAMES.get(classe, "Classe inconnue")
        st.success(f"Classe prédite : **{classe_nom}** (index : {classe})")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
        
        proba = np.max(predictions)
        st.write(f"Confiance du modèle : **{proba:.2%}**")