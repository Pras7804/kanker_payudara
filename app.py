import streamlit as st
import joblib
import numpy as np

# Load pipeline model
model = joblib.load('knn_model.pkl')

# 30 fitur asli sebelum seleksi (kolom dari df_raw tanpa 'Diagnosis')
feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1',
    'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
    'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2',
    'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3',
    'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'
]

# Judul halaman
st.set_page_config(page_title="Prediksi Kanker Payudara", layout="centered")
st.title("🩺 Prediksi Kanker Payudara")
st.markdown("""
Aplikasi ini menggunakan model **K-Nearest Neighbors (KNN)** untuk memprediksi apakah suatu tumor bersifat **Jinak (Benign)** atau **Ganas (Malignant)** berdasarkan 30 fitur medis dari dataset WDBC.
""")

# Input fitur dari pengguna
st.subheader("Masukkan nilai untuk masing-masing fitur:")
user_input = []

with st.form("input_form"):
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
        user_input.append(value)
    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi saat tombol ditekan
if submitted:
    try:
        prediction = model.predict([user_input])[0]
        hasil = "🟥 Ganas (Malignant)" if prediction == 1 else "🟩 Jinak (Benign)"
        st.markdown(f"### Hasil Prediksi: {hasil}")
        st.success("Prediksi berhasil dijalankan.")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# Footer
st.markdown("---")
st.caption("Model ini dibangun menggunakan dataset Wisconsin Diagnostic Breast Cancer (UCI Repository).")
