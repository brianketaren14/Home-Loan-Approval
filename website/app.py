import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Config / Set Page Settings
st.set_page_config(
    page_title="Sistem Persetujuan Pinjaman (Loan Approval)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = Path("model") / "model.pkl"
SCALER_PATH = Path("model") / "scaler.pkl"
SELECTED_FEATURES_PATH = Path("model") / "selected_features.pkl"

# Load models and resources
@st.cache_resource
def load_resources():
    try:
        if not MODEL_PATH.exists():
            return None, None, None, f"File model tidak ditemukan di: {MODEL_PATH.absolute()}"
        if not SCALER_PATH.exists():
            return None, None, None, f"File scaler tidak ditemukan di: {SCALER_PATH.absolute()}"
        if not SELECTED_FEATURES_PATH.exists():
            return None, None, None, f"File selected_features tidak ditemukan di: {SELECTED_FEATURES_PATH.absolute()}"

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(SELECTED_FEATURES_PATH, 'rb') as f:
            selected_features = pickle.load(f)
        
        return model, scaler, selected_features, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, selected_features, load_error = load_resources()


# Main Title Header
st.markdown("<div class='glow-header'>SISTEM PREDIKSI PERSETUJUAN PINJAMAN</div>", unsafe_allow_html=True)
st.markdown("Masukkan data pemohon secara lengkap untuk menganalisis kelayakan kredit berdasarkan model Machine Learning.")
st.markdown("---")

if load_error:
    st.error(f"⚠️ **Error Memuat Resource:** {load_error}")
    st.warning("Pastikan file model.pkl, scaler.pkl, dan selected_features.pkl sudah diletakkan di dalam folder `model/`.")
else:
    # 3 Column Form Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("👤 Data Pribadi")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Status Pernikahan (Married)", ["No", "Yes", "Unknown"])
        dependents = st.selectbox("Jumlah Tanggungan (Dependents)", ["0", "1", "2", "3+", "Unknown"])
        education = st.selectbox("Pendidikan (Education)", ["Graduate", "Not Graduate"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("💼 Data Finansial")
        self_employed = st.selectbox("Wiraswasta (Self Employed)", ["No", "Yes"])
        applicant_income = st.number_input("Pendapatan Pemohon (Applicant Income / $)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Pendapatan Penjamin (Co-applicant Income / $)", min_value=0.0, value=0.0, step=500.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("💳 Data Pinjaman & Area")
        loan_amount = st.number_input("Jumlah Pinjaman (Loan Amount / Ribu $)", min_value=0.0, value=120.0, step=10.0)
        loan_amount_term = st.selectbox("Jangka Waktu Pinjaman (Loan Amount Term / Bulan)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
        credit_history = st.selectbox("Riwayat Kredit (Credit History)", [1.0, 0.0])
        property_area = st.selectbox("Area Properti (Property Area)", ["Urban", "Semiurban", "Rural"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Predict Button
    if st.button("MULAI PROSES ANALISIS DATA"):
        # 1. Feature scaling on numerical inputs
        # Create DataFrame with the exact numeric names expected by scaler
        numeric_df = pd.DataFrame(
            [[applicant_income, coapplicant_income, loan_amount, loan_amount_term]],
            columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        )
        
        # Fitur scaling using RobustScaler
        scaled_numeric = scaler.transform(numeric_df)
        
        scaled_app_income = scaled_numeric[0][0]
        scaled_coapp_income = scaled_numeric[0][1]
        scaled_loan_amount = scaled_numeric[0][2]
        scaled_loan_term = scaled_numeric[0][3]

        # 2. Build the full list of 19 encoded features (already mapped)
        encoded_features = {
            'ApplicantIncome': scaled_app_income,
            'CoapplicantIncome': scaled_coapp_income,
            'LoanAmount': scaled_loan_amount,
            'Loan_Amount_Term': scaled_loan_term,
            'Gender_Male': 1 if gender == "Male" else 0,
            'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
            'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
            'Credit_History_Yes': 1 if credit_history == 1.0 else 0,
            'Married_No': 1 if married == "No" else 0,
            'Married_Yes': 1 if married == "Yes" else 0,
            'Married_unknown': 1 if married == "Unknown" else 0,
            'Dependents_0': 1 if dependents == "0" else 0,
            'Dependents_1': 1 if dependents == "1" else 0,
            'Dependents_2': 1 if dependents == "2" else 0,
            'Dependents_3+': 1 if dependents == "3+" else 0,
            'Dependents_unknown': 1 if dependents == "Unknown" else 0,
            'Property_Area_Rural': 1 if property_area == "Rural" else 0,
            'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
            'Property_Area_Urban': 1 if property_area == "Urban" else 0
        }

        # Convert dictionary to DataFrame
        df_encoded = pd.DataFrame([encoded_features])

        # 3. Cut dimensions using selected_features
        df_selected = df_encoded[selected_features]

        # Run Prediction
        prediction = model.predict(df_selected)
        
        # Display Prediction Result Card
        st.subheader("🎯 HASIL ANALISIS KELAYAKAN")
        
        if prediction[0] == 1:
            st.markdown("""
                <div class='result-approved'>
                    <h2>🎉 PINJAMAN DISETUJUI</h2>
                    <p style="font-size: 1.2rem; margin: 0;">Sistem mengonfirmasi bahwa profil pemohon memenuhi kriteria kelayakan kredit. [STATUS: APPROVED]</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-rejected'>
                    <h2>❌ PINJAMAN DITOLAK</h2>
                    <p style="font-size: 1.2rem; margin: 0;">Profil pemohon tidak memenuhi standar kelayakan kredit minimum. [STATUS: REJECTED]</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for details
        tab1, tab2, tab3 = st.tabs(["📝 Input Data & Hasil Scaling", "🔢 19 Fitur Ter-encode", "🔮 Fitur Pilihan Model"])
        
        with tab1:
            st.write("### Data Numerik Awal vs Ter-scaling")
            df_compare = pd.DataFrame({
                "Nama Fitur": ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"],
                "Nilai Input Asli": [applicant_income, coapplicant_income, loan_amount, loan_amount_term],
                "Nilai Setelah Scaling (RobustScaler)": [scaled_app_income, scaled_coapp_income, scaled_loan_amount, scaled_loan_term]
            })
            st.dataframe(df_compare, use_container_width=True)

        with tab2:
            st.write("### 19 Fitur Hasil Encoding Lengkap")
            st.dataframe(df_encoded, use_container_width=True)

        with tab3:
            st.write("### 10 Fitur Terpilih yang Dikirimkan ke Model (Dimension Cut)")
            st.dataframe(df_selected, use_container_width=True)
