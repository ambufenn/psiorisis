import streamlit as st
from google import genai
from google.genai import types
import numpy as np
from datetime import datetime
import random 

# --- 0. Setup Klien Gemini & Konfigurasi ---
# PENTING: Ganti "YOUR_GEMINI_API_KEY" dengan kunci Anda.
try:
    # Dalam lingkungan Streamlit Cloud, Anda harus menyimpan kunci di st.secrets
    API_KEY = "YOUR_GEMINI_API_KEY" # Gantilah ini!
    client = genai.Client(api_key=API_KEY)
    MODEL_FLASH = 'gemini-2.5-flash'
except Exception:
    client = None
    st.error("⚠️ Kunci API Gemini tidak valid atau tidak ditemukan. Fitur AI Summary/Coaching akan dinonaktifkan.")

# --- 1. Fungsi ML dan AI Inti ---

def simulate_predict_flare_risk(data):
    """
    [PPS: Modul Kuratif] SIMULASI ML: Prediksi risiko Flare (Stres/Penyakit).
    """
    pain_norm = data.get('pain_score', 0) / 10
    stress_norm = data.get('stress_score', 0) / 10
    adherence_impact = 1.0 - data.get('med_adherence', 1.0) 
    hrv_impact = 1.0 - data.get('hrv_avg', 0.8) 
    
    # Formula risiko yang menempatkan stres (stress_norm, hrv_impact) sebagai bobot tinggi
    risk_score = (pain_norm * 0.2) + (stress_norm * 0.3) + (adherence_impact * 0.3) + (hrv_impact * 0.2)
    
    return min(risk_score * 0.9, 0.99)

def generate_stress_coaching_gemini(user_mood, hrv_status, time_of_day):
    """
    [PPS: Modul Kuratif] Fitur: Mindfulness & Relaksasi Coach (Real-Time).
    """
    if not client: return "⚠️ Fitur Gemini dinonaktifkan."

    system_instruction = """
    Anda adalah Stress Management Coach yang Empati untuk pasien Psoriasis Arthritis.
    Berikan instruksi teknik relaksasi adaptif dan jelaskan relevansi sesi ini (1 paragraf).
    Sajikan instruksi dalam 3-5 langkah yang mudah diikuti.
    """
    prompt = f"Pasien melaporkan mood: '{user_mood}'. Status HRV objektif: {hrv_status}. Waktu: {time_of_day}."
    
    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
    with st.spinner("Gemini sedang menyiapkan sesi coaching personal..."):
        try:
            response = client.models.generate_content(model=MODEL_FLASH, contents=prompt, config=config)
            return response.text
        except Exception:
            return "Error dalam koneksi atau respons Gemini."

def generate_clinician_summary(user_logs):
    """
    [PPS: Modul Integrasi] Fitur: Ringkasan Data Longitudinal untuk Dokter (Gemini).
    """
    if not client: return "⚠️ Fitur Gemini dinonaktifkan."

    latest_logs = user_logs[-7:]
    
    summary_prompt = f"""
    Menganalisis log kesehatan pasien selama {len(user_logs)} entri.
    Data Log Terbaru: {latest_logs}
    Tolong buat ringkasan status pasien (max 3 paragraf) untuk Reumatolog, menyoroti:
    1. Tren aktivitas penyakit (Nyeri/Kekakuan).
    2. Tingkat kepatuhan obat dan efek samping.
    3. Pengaruh Stres (HRV) terhadap gejala dalam seminggu terakhir.
    """

    config = types.GenerateContentConfig(system_instruction="Anda adalah Asisten Klinis AI. Buat ringkasan data pasien yang terstruktur dan objektif.", temperature=0.3)
    
    with st.spinner("Menganalisis data log dan menyusun ringkasan klinis..."):
        try:
            response = client.models.generate_content(model=MODEL_FLASH, contents=summary_prompt, config=config)
            return response.text
        except Exception:
            return "Failed to generate AI summary."

# --- 2. Session State & Input Data ---

if 'user_log' not in st.session_state:
    st.session_state.user_log = []

st.set_page_config(layout="wide", page_title="PsA Intelligence System")
st.title("🛡️ Gemini PsA Intelligence System Dashboard")
st.caption("Implementasi Fungsionalitas PPS: Logika ML dan Gemini Dijalankan Langsung.")
st.markdown("---")

# --- 2.1 Modul Preventif/Kuratif: Input Data Harian ---
st.header("1. Input Data Harian & Log Gejala (Simulasi Aplikasi Mobile)")

with st.form("log_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pain_score = st.slider("Nyeri Sendi (Skala 0-10)", 0, 10, 5, key='pain')
        stiffness_score = st.slider("Kekakuan Pagi (Skala 0-10)", 0, 10, 5, key='stiffness')

    with col2:
        stress_score = st.slider("Stres Subjektif (Skala 0-10)", 0, 10, 6, key='stress')
        hrv_avg = st.slider("HRV Rata-rata (0.0=Buruk, 1.0=Baik)", 0.0, 1.0, 0.65, 0.05, key='hrv')

    with col3:
        med_adherence = st.slider("Kepatuhan Obat (%)", 0, 100, 100, 10, key='med')
        video_rehab_status = st.selectbox(
            "Feedback Vision/Rehab (Simulasi):", 
            ["Good", "Poor_Posture", "Fatigue"], 
            key='rehab_status'
        )

    submitted = st.form_submit_button("Simpan Log Hari Ini & Analisis", type="primary")

    if submitted:
        new_log = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'pain_score': pain_score,
            'stiffness_score': stiffness_score,
            'stress_score': stress_score,
            'hrv_avg': hrv_avg,
            'med_adherence': med_adherence / 100.0,
            'video_rehab_status': video_rehab_status
        }
        st.session_state.user_log.append(new_log)
        st.success(f"Log berhasil disimpan. Total {len(st.session_state.user_log)} entri.")

st.markdown("---")

# --- 3. Modul Kuratif: Prediksi & Alert ML ---
st.header("2. Prediksi Risiko Flare (ML/Alert)")

if st.session_state.user_log:
    latest_data = st.session_state.user_log[-1]
    
    # Panggil fungsi simulasi ML (Predictive Tool)
    risk_prob = simulate_predict_flare_risk(latest_data)
    prob_percent = risk_prob * 100
    
    if risk_prob > 0.75:
        st.error(f"🚨 ALERT TINGGI: Risiko Flare ({prob_percent:.1f}%). Pemicu utama: Stres/HRV. {random.choice(['Segera lakukan intervensi stress.', 'Hubungi klinik untuk triage.'])}")
    elif risk_prob > 0.55:
        st.warning(f"⚠️ RISIKO MODERAT: Risiko Flare sedang ({prob_percent:.1f}%). Fokus pada istirahat & ikuti sesi coaching.")
    else:
        st.success(f"Risiko Flare Rendah ({prob_percent:.1f}%).")

    st.caption("*(Simulasi ML: Menggunakan Nyeri, Stres, HRV, dan Kepatuhan Obat)*")
else:
    st.info("Masukkan log harian untuk menjalankan prediksi.")

st.markdown("---")

# --- 4. Modul Intervensi Mandiri & Rehabilitasi ---
st.header("3. Intervensi Cerdas (Gemini Coach & Rehabilitasi)")

col_i1, col_i2 = st.columns(2)

with col_i1:
    st.subheader("Mindfulness & Relaksasi Coach")
    if st.session_state.user_log and st.button("Mulai Sesi Stress Coaching (Gemini)", key='start_coaching', type='secondary'):
        
        current_mood = "Tegang dan butuh menenangkan pikiran"
        hrv_status_text = "HRV rendah (di bawah 0.4)" if latest_data['hrv_avg'] < 0.4 else "HRV normal"
        
        coaching_result = generate_stress_coaching_gemini(
            user_mood=current_mood,
            hrv_status=hrv_status_text,
            time_of_day=datetime.now().strftime("%H:%M")
        )
        
        st.markdown(f"**Hasil Coaching:**\n{coaching_result}")

with col_i2:
    st.subheader("Activity Pacing & Fisioterapi")
    if st.session_state.user_log:
        
        # [PPS: Modul Rehabilitasi] Logika Activity Pacing Coach
        if latest_data['hrv_avg'] < 0.5 or latest_data['stress_score'] > 7:
            pacing_advice = "🚨 **Activity Pacing Coach:** Kelelahan terdeteksi (HRV rendah). Kurangi intensitas latihan 50% dan prioritaskan istirahat 30 menit."
        else:
            pacing_advice = "**Activity Pacing Coach:** Energi stabil. Pertahankan jadwal latihan Anda."
            
        st.warning(pacing_advice)
        
        # [PPS: Modul Rehabilitasi] Simulasi Computer Vision Feedback
        if latest_data['video_rehab_status'] == 'Poor_Posture':
            st.error("**Feedback Vision:** Postur latihan buruk terdeteksi. Harap jaga punggung lurus.")
        elif latest_data['video_rehab_status'] == 'Fatigue':
            st.warning("**Feedback Vision:** Kelelahan terdeteksi saat latihan. Lakukan istirahat aktif.")
        else:
            st.success("Feedback Vision: Latihan diselesaikan dengan postur yang baik.")

st.markdown("---")

# --- 5. Modul Integrasi Sistem (Dashboard Klinis) ---
st.header("4. Dashboard Klinis & Integrasi EHR (Untuk Dokter)")

if st.session_state.user_log:
    
    st.subheader("Ringkasan Klinis AI (Gemini)")
    if st.button("Generate AI Summary untuk Reumatolog", key='generate_summary', type='primary'):
        ai_summary = generate_clinician_summary(st.session_state.user_log)
        st.info("Ringkasan ini ditujukan untuk tim klinis dan dapat diunggah ke EHR.")
        st.markdown(ai_summary)

    st.subheader("Prediksi Terapi Biologics (Simulasi ML)")
    if st.button("Prediksi Respons Terapi Biologics", key='predict_drug'):
        
        # [PPS: Modul Kuratif] Fitur: Predictive Therapy Tool
        drug_a_prob = random.uniform(0.6, 0.9)
        drug_b_prob = random.uniform(0.5, 0.8)
        
        st.code(
            f"Probabilitas Remisi Drug A (TNF Inhibitor): {drug_a_prob:.2f}\n"
            f"Probabilitas Remisi Drug B (IL-17 Inhibitor): {drug_b_prob:.2f}\n"
            "\n**Rekomendasi ML:** Drug A direkomendasikan karena rasio efikasi-biaya yang lebih baik (Simulasi data cohort Singapura)."
        )

# Tampilkan log data mentah
st.sidebar.title("Data Log Mentah")
st.sidebar.json(st.session_state.user_log)
