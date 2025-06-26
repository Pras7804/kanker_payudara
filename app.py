# app.py
import streamlit as st
import joblib

# Load model pipeline
model = joblib.load('knn_model.pkl')

# Fitur input dari user sesuai dengan hasil rename
feature_names = [
    'radius_mean',
    'perimeter_mean',
    'area_mean',
    'concavity_mean',
    'concave_points_mean',
    'radius_worst',
    'perimeter_worst',
    'area_worst',
    'concavity_worst',
    'concave_points_worst'
]

st.set_page_config(page_title="Prediksi Kanker Payudara", layout="centered")
st.title("🩺 Prediksi Kanker Payudara")
st.markdown("Masukkan nilai-nilai fitur medis berikut untuk memprediksi apakah tumor bersifat **Jinak (Benign)** atau **Ganas (Malignant)**.")

user_input = []

with st.form("form_input"):
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
        user_input.append(value)
    submit = st.form_submit_button("🔍 Prediksi")

if submit:
    prediction = model.predict([user_input])[0]
    hasil = "🟥 Ganas (Malignant)" if prediction == 1 else "🟩 Jinak (Benign)"
    st.markdown(f"### Hasil Prediksi: {hasil}")
    st.success("Prediksi berhasil dijalankan.")

st.markdown("---")
st.caption("Model ini menggunakan 10 fitur hasil seleksi dari dataset Wisconsin Diagnostic Breast Cancer.")
