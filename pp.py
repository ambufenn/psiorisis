import streamlit as st
from google import genai
from google.genai import types
import numpy as np
import random 

# --- 0. Setup Klien Gemini & Konfigurasi ---
try:
    # Ambil kunci API dari Streamlit Secrets
    API_KEY = st.secrets["gemini_api_key"]
    client = genai.Client(api_key=API_KEY)
    MODEL_FLASH = 'gemini-2.5-flash'
except Exception:
    st.error("Kunci API Gemini tidak ditemukan di st.secrets['gemini_api_key']. Beberapa fitur AI dinonaktifkan.")
    client = None

# --- FUNGSI AI DAN ML SIMULASI (dari pps.py dan gemini_prompter.py) ---

def simulate_predict_flare_risk(data):
    """
    Simulasi ML: Memprediksi risiko flare berdasarkan Nyeri, Stres, dan HRV.
    Data input: Dict dengan keys: 'pain_score', 'stress_score', 'hrv_avg'.
    """
    # Normalisasi data
    pain_norm = data.get('pain_score', 0) / 10
    stress_norm = data.get('stress_score', 0) / 10
    
    # Dampak HRV rendah (0.0 sangat rendah, 1.0 normal)
    hrv_impact = 1.0 - data.get('hrv_avg', 0.8) 
    
    # Formula simulasi risiko
    risk_score = (pain_norm * 0.3) + (stress_norm * 0.4) + (hrv_impact * 0.3)
    
    return min(risk_score * 0.8, 0.95) # Hasil probabilitas (0.0 hingga 0.95)


def generate_stress_coaching_gemini(user_mood, hrv_status, time_of_day):
    """Fitur: Mindfulness & Relaksasi Coach (Gemini)"""
    if not client: return "⚠️ Fitur Gemini dinonaktifkan."

    system_instruction = """
    Anda adalah Stress Management Coach yang Empati dan suportif untuk pasien autoimun.
    Berikan instruksi teknik relaksasi (pernapasan/mindfulness) berbasis data real-time,
    sertakan penjelasan singkat (1 paragraf) mengapa sesi ini penting.
    Sajikan instruksi dalam 3-5 langkah yang mudah diikuti.
    """

    prompt = f"Pasien melaporkan mood: '{user_mood}'. Status HRV objektif: {hrv_status}. Waktu: {time_of_day}."
    
    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
    
    with st.spinner("Gemini sedang menyiapkan sesi coaching personal..."):
        try:
            response = client.models.generate_content(model=MODEL_FLASH, contents=prompt, config=config)
            return response.text
        except Exception as e:
            return f"Error AI: Gagal menghasilkan coaching. {e}"

def summarize_for_clinician_gemini(log_data):
    """Fitur: Ringkasan Data Longitudinal untuk Dokter (Gemini)"""
    if not client: return "⚠️ Fitur Gemini dinonaktifkan."
    
    # Hanya kirim 5 log terakhir agar tidak terlalu panjang
    latest_logs = log_data[-5:]

    summary_prompt = f"""
    Menganalisis log kesehatan pasien selama {len(log_data)} hari/minggu.
    Log data: {latest_logs}
    Tolong buat ringkasan status pasien yang ringkas (maksimal 3 paragraf pendek)
    untuk Reumatolog, menyoroti:
    1. Tren nyeri sendi utama (naik/turun).
    2. Tingkat Stres (HRV dan Skor Subjektif).
    3. Hubungan antara Stres dan Flare jika terlihat.
    """

    config = types.GenerateContentConfig(system_instruction="Anda adalah Asisten Klinis AI. Buat ringkasan data pasien yang terstruktur dan objektif.", temperature=0.3)
    
    with st.spinner("Menganalisis data log dan menyusun ringkasan klinis..."):
        try:
            response = client.models.generate_content(model=MODEL_FLASH, contents=summary_prompt, config=config)
            return response.text
        except Exception as e:
            return f"Error AI: Gagal membuat ringkasan. {e}"


# --- Session State untuk menyimpan Log Pasien ---
if 'user_log' not in st.session_state:
    st.session_state.user_log = []

# --- Halaman Streamlit Utama ---
st.set_page_config(layout="wide", page_title="PsA Intelligence System (Streamlit)")
st.title("🛡️ Gemini PsA Intelligence System")
st.caption("Simulasi Dashboard Klinik & Intervensi Mandiri (Berdasarkan Konsep PPS Singapura)")
st.markdown("---")

# --- 1. Modul Preventif/Kuratif: Input Data Harian ---
st.header("1. Input Data Harian & Log Gejala")
st.markdown("Masukkan data harian Pasien (simulasi dari aplikasi mobile/wearable).")

col1, col2, col3 = st.columns(3)

with col1:
    pain_score = st.slider("Nyeri Sendi (Skala 0-10)", 0, 10, 5, key='pain')
    stiffness_score = st.slider("Kekakuan Pagi (Skala 0-10)", 0, 10, 5, key='stiffness')

with col2:
    stress_score = st.slider("Stres Subjektif (Skala 0-10)", 0, 10, 6, key='stress')
    hrv_avg = st.slider("HRV Rata-rata (Simulasi Wearable: 0.0=Buruk, 1.0=Baik)", 0.0, 1.0, 0.65, 0.05, key='hrv')

with col3:
    med_adherence = st.slider("Kepatuhan Obat (%)", 0, 100, 100, 10, key='med')
    kebutuhan_coach = st.selectbox("Kebutuhan *Coaching* Hari Ini:", ["Tidak", "Relaksasi Cepat", "Saran Pacing"], key='coach_need')


if st.button("Simpan Log Hari Ini & Analisis", type="primary"):
    new_log = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'pain_score': pain_score,
        'stiffness_score': stiffness_score,
        'stress_score': stress_score,
        'hrv_avg': hrv_avg,
        'med_adherence': med_adherence / 100.0
    }
    st.session_state.user_log.append(new_log)
    st.success(f"Log berhasil disimpan. Total {len(st.session_state.user_log)} entri.")

st.markdown("---")

# --- 2. Modul Kuratif: Prediksi & Alert ML ---
st.header("2. Prediksi Risiko Flare (ML/Alert)")

if st.session_state.user_log:
    latest_data = st.session_state.user_log[-1]
    
    # Panggil fungsi simulasi ML
    risk_prob = simulate_predict_flare_risk(latest_data)
    prob_percent = risk_prob * 100
    
    st.metric("Probabilitas Risiko Flare (7 hari)", f"{prob_percent:.1f}%")

    if risk_prob > 0.75:
        st.error(f"🚨 ALERT TINGGI: Risiko Flare sangat tinggi ({prob_percent:.1f}%). Pemicu utama: Stres Tinggi ({latest_data['stress_score']}) dan HRV Rendah ({latest_data['hrv_avg']:.2f}).")
    elif risk_prob > 0.55:
        st.warning(f"⚠️ RISIKO MODERAT: Risiko Flare sedang ({prob_percent:.1f}%). Lakukan sesi coaching segera.")
    else:
        st.success(f"Risiko Flare Rendah. Lanjutkan monitoring.")
    
    st.caption("*(Simulasi ML: Menggunakan Pain, Stres, dan HRV untuk prediksi)*")
else:
    st.info("Masukkan log harian untuk menjalankan prediksi.")

st.markdown("---")

# --- 3. Modul Intervensi Mandiri (Gemini Coaching) ---
st.header("3. Intervensi Cerdas (Gemini Coach)")

if kebutuhan_coach != "Tidak" and st.button("Mulai Sesi Relaksasi Cerdas (Gemini)", key='start_coaching', type='secondary'):
    
    mood_input = "Tegang dan butuh menenangkan pikiran" if kebutuhan_coach == "Relaksasi Cepat" else "Lelah dan ingin mengatur jadwal"
    
    # Ambil status terbaru dari input untuk personalisasi
    hrv_status_text = "HRV sangat rendah, sinyal istirahat" if hrv_avg < 0.4 else "HRV normal, coba latihan ringan"
    
    coaching_result = generate_stress_coaching_gemini(
        user_mood=mood_input,
        hrv_status=hrv_status_text,
        time_of_day=datetime.now().strftime("%H:%M")
    )
    
    st.markdown("### 🧘 Hasil Coaching Gemini:")
    st.markdown(coaching_result)

st.markdown("---")

# --- 4. Modul Integrasi Klinis (Ringkasan EHR) ---
st.header("4. Ringkasan Klinis untuk Reumatolog")

if st.session_state.user_log:
    if st.button("Generate AI Summary (Gemini)", key='generate_summary', type='primary'):
        ai_summary = summarize_for_clinician_gemini(st.session_state.user_log)
        
        st.info("Ringkasan ini ditujukan untuk tim klinis (Reumatolog/Sp. KFR) dan dapat dikirimkan ke EHR.")
        st.markdown(ai_summary)
else:
    st.info("Kumpulkan data log minimal 2 hari untuk membuat Ringkasan AI yang berarti.")
