Mohon maaf, terjadi kesalahan di mana AI model menghasilkan gambar sebagai respons atas permintaan Anda. Saya tidak berniat untuk mengulang kesalahan tersebut.

Anda meminta pembaruan pada kode Streamlit (`streamlit_app_final_v3.py`) untuk menambahkan tiga fitur penting:

1.  **Diagnosis Potensial** (Modul Preventif).
2.  **Upload Foto** untuk Cek Psoriasis (Modul Preventif/Skrining).
3.  **Data History Kesehatan** kita.

Saya akan membuat satu tab baru (`tab_skrining`) untuk menampung fitur diagnostik potensial dan upload foto, serta menambahkan tampilan riwayat kesehatan yang lebih detail di Tab Dashboard.

Berikut adalah kode yang **sudah diperbarui** dan **lengkap**:

-----

## 💻 `streamlit_app_final_v4.py`: Gemini PsA Intelligence System (Final + Diagnostik)

```python
import streamlit as st
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from datetime import datetime
import random 
from PIL import Image
import io

# --- 0. Setup Klien Gemini & Konfigurasi ---
try:
    # PENTING: Ganti dengan kunci API Anda. Di Streamlit Cloud, gunakan st.secrets
    API_KEY = "YOUR_GEMINI_API_KEY" # <<< GANTI DENGAN KUNCI ASLI ANDA
    client = genai.Client(api_key=API_KEY)
    MODEL_FLASH = 'gemini-2.5-flash'
    MODEL_PRO_VISION = 'gemini-2.5-flash' # Model untuk analisis gambar
except Exception:
    client = None
    # Fitur AI akan dinonaktifkan jika kunci tidak valid
    
# --- 1. Fungsi AI dan ML Inti ---

def simulate_predict_flare_risk(data):
    """[ML] Prediksi risiko Flare (Modul Kuratif)."""
    pain_norm = data.get('pain_score', 0) / 10
    stress_norm = data.get('stress_score', 0) / 10
    hrv_impact = 1.0 - data.get('hrv_avg', 0.8) 
    risk_score = (pain_norm * 0.25) + (stress_norm * 0.35) + (hrv_impact * 0.4) 
    return min(risk_score * 0.9, 0.99)

def generate_stress_coaching_gemini(user_mood, hrv_status, time_of_day):
    """[Gemini] Fitur: Mindfulness & Relaksasi Coach (Modul Kuratif)."""
    if not client: return "⚠️ Layanan Gemini dinonaktifkan."
    system_instruction = "Anda adalah Stress Management Coach yang Empati untuk pasien PsA. Berikan instruksi relaksasi adaptif, jelaskan relevansi sesi, dan sajikan instruksi dalam 3-5 langkah."
    prompt = f"Pasien melaporkan mood: '{user_mood}'. Status HRV objektif: {hrv_status}. Waktu: {time_of_day}."
    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
    try:
        response = client.models.generate_content(model=MODEL_FLASH, contents=prompt, config=config)
        return response.text
    except Exception:
        return "Error dalam koneksi atau respons Gemini."

def analyze_skin_photo(image_file):
    """[Gemini Vision] Analisis Foto Kulit untuk Skrining Psoriasis/PsA (Modul Preventif)."""
    if not client: return "⚠️ Layanan Gemini dinonaktifkan."
    
    img = Image.open(image_file)
    
    prompt = (
        "Analisis gambar kulit ini. Apakah terdapat ciri-ciri lesi Psoriasis "
        "(plak merah tebal, sisik perak)? Berikan diagnosis potensial (contoh: 'Kemungkinan Psoriasis', 'Tidak Khas PsA', 'Pasti Dermatitis'). "
        "Tekankan bahwa ini adalah Skrining AI dan BUKAN diagnosis medis final."
    )
    
    contents = [img, prompt]
    
    with st.spinner("Menganalisis foto dengan Gemini Vision..."):
        try:
            response = client.models.generate_content(model=MODEL_PRO_VISION, contents=contents)
            return response.text
        except Exception as e:
            return f"Error Vision: Gagal menganalisis gambar. {e}"


def generate_clinician_summary(user_logs, user_history):
    """[Gemini] Ringkasan Data Longitudinal (Modul Integrasi)."""
    if not client: return "⚠️ Layanan Gemini dinonaktifkan."
    
    latest_logs = user_logs[-7:]
    
    summary_prompt = f"""
    Buat ringkasan status pasien (max 3 paragraf) untuk Reumatolog, menyoroti:
    1. Riwayat Kesehatan Utama Pasien: {user_history}.
    2. Tren aktivitas penyakit (Nyeri/Kekakuan) dari {len(user_logs)} entri.
    3. Pengaruh Stres/HRV terhadap gejala dalam seminggu terakhir.
    """
    config = types.GenerateContentConfig(system_instruction="Anda adalah Asisten Klinis AI. Buat ringkasan data pasien yang terstruktur, objektif, dan menggabungkan riwayat kesehatan.", temperature=0.3)
    
    try:
        response = client.models.generate_content(model=MODEL_FLASH, contents=summary_prompt, config=config)
        return response.text
    except Exception:
        return "Failed to generate AI summary."

def generate_chatbot_response(prompt):
    """[Gemini] Fitur: Asisten Kesehatan PsA (Character Bot)."""
    if not client: return "Layanan Chatbot dinonaktifkan tanpa API Key."
    
    system_instruction = "Anda adalah Asisten Kesehatan AI yang ramah, berempati, dan berpengetahuan khusus tentang Psoriasis Arthritis (PsA) dan manajemen gaya hidup. Jawab pertanyaan dengan akurat, tetapi selalu tegaskan bahwa Anda BUKAN dokter dan tidak dapat memberikan saran dosis obat atau diagnosis darurat."
    
    history = [types.Content(role=m["role"], parts=[types.Part.from_text(m["content"])]) for m in st.session_state.messages]
    history.append(types.Content(role="user", parts=[types.Part.from_text(prompt)]))

    config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.5)

    try:
        response = client.models.generate_content(model=MODEL_FLASH, contents=history, config=config)
        return response.text
    except Exception:
        return "Terjadi kesalahan saat memproses permintaan Chatbot."

# --- 2. Session State & Setup Halaman ---

if 'user_log' not in st.session_state:
    st.session_state.user_log = [
        {'timestamp': '2025-12-01 10:00', 'pain_score': 3, 'hrv_avg': 0.7, 'stress_score': 3, 'med_adherence': 1.0, 'video_rehab_status': 'Good'},
        {'timestamp': '2025-12-03 10:00', 'pain_score': 5, 'hrv_avg': 0.4, 'stress_score': 7, 'med_adherence': 0.8, 'video_rehab_status': 'Poor_Posture'},
    ]

if 'user_history' not in st.session_state:
    st.session_state.user_history = {
        'diagnosis': 'Psoriasis Vulgaris (2020)',
        'komorbiditas': 'Obesitas',
        'obat_saat_ini': 'Methotrexate 15mg/minggu'
    }

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Halo! Saya Asisten Kesehatan PsA Anda. Tanyakan apa saja tentang Psoriasis Arthritis dan gaya hidup."}]


st.set_page_config(layout="wide", page_title="PsA Intelligence System")
st.title("🛡️ Gemini PsA Intelligence System")
st.caption("Implementasi Kode Lengkap PPS: Skrining, Prediksi, Intervensi, dan Integrasi.")

df_log = pd.DataFrame(st.session_state.user_log)
if not df_log.empty:
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
    df_log = df_log.sort_values('timestamp')

# --- 3. Tata Letak Tab Baru ---

tab_log, tab_skrining, tab_intervensi, tab_dashboard, tab_chatbot = st.tabs(["📝 Log & Alert", "📸 Skrining Potensial & Foto", "🧠 Intervensi & Rehab", "👨‍⚕️ Dashboard Klinis", "💬 Asisten PsA"])


# =========================================================================
# === TAB 1: LOG HARIAN & ALERT
# =========================================================================

with tab_log:
    st.header("Input Data Pasien & Prediksi Risiko")

    with st.form("log_form"):
        col_input_1, col_input_2 = st.columns(2)

        with col_input_1:
            pain_score = st.slider("Nyeri Sendi (0-10)", 0, 10, 5, key='pain')
            med_adherence = st.slider("Kepatuhan Obat (%)", 0, 100, 100, 10, key='med')
            video_rehab_status = st.selectbox(
                "Feedback Vision/Rehab (Simulasi):", 
                ["Good", "Poor_Posture", "Fatigue"], 
                key='rehab_status'
            )

        with col_input_2:
            stress_score = st.slider("Stres Subjektif (0-10)", 0, 10, 6, key='stress')
            hrv_avg = st.slider("HRV Rata-rata (0.0=Buruk, 1.0=Baik)", 0.0, 1.0, 0.65, 0.05, key='hrv')
            stiffness_score = st.slider("Kekakuan Pagi (0-10)", 0, 10, 5, key='stiffness')


        submitted = st.form_submit_button("Simpan Log Baru & Hitung Risiko", type="primary")

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
            st.rerun() 

    st.subheader("⚠️ Hasil Prediksi Risiko Flare (Anomaly Alert)")
    if st.session_state.user_log:
        latest_data = st.session_state.user_log[-1]
        risk_prob = simulate_predict_flare_risk(latest_data)
        prob_percent = risk_prob * 100
        
        col_alert_1, col_alert_2, col_alert_3 = st.columns(3)
        col_alert_1.metric("Probabilitas Flare (7 hari)", f"{prob_percent:.1f}%")
        col_alert_2.metric("HRV Terbaru", f"{latest_data['hrv_avg']:.2f}")

        if risk_prob > 0.75:
            st.error(f"🚨 ALERT TINGGI: Risiko Flare sangat tinggi. Tindakan: Segera ke tab Intervensi.")
        elif risk_prob > 0.55:
            st.warning(f"⚠️ RISIKO MODERAT: Fokus pada manajemen stres.")
        else:
            st.success(f"Risiko Flare Rendah.")


# =========================================================================
# === TAB 2: SKRINING POTENSIAL & FOTO (Modul Preventif)
# =========================================================================

with tab_skrining:
    st.header("Diagnosis Potensial & Skrining Kulit AI")

    col_diag_1, col_diag_2 = st.columns(2)

    with col_diag_1:
        st.subheader("1. Diagnosis Potensial (PEST/TOPAS)")
        st.markdown(f"**Riwayat Kesehatan Terkini:**")
        st.write(f"**Diagnosis Primer:** {st.session_state.user_history['diagnosis']}")
        st.write(f"**Komorbiditas:** {st.session_state.user_history['komorbiditas']}")
        st.write(f"**Obat Aktif:** {st.session_state.user_history['obat_saat_ini']}")
        
        # Simulasi PEST/Skrining PsA
        st.subheader("Skrining Risiko PsA")
        if st.session_state.user_log and st.button("Hitung Skor Risiko PsA (PEST/TOPAS)", type='secondary'):
            # Logika Simulasi PEST: Jika ada Nyeri & Kekakuan, risiko PsA tinggi
            if st.session_state.user_log[-1]['pain_score'] >= 5 and st.session_state.user_log[-1]['stiffness_score'] >= 5:
                st.error("⚠️ RISIKO PSORIATIC ARTHRITIS TINGGI. Disarankan Rujukan Reumatolog.")
                st.info("Skor PEST (Simulasi) > 3. Ada indikasi nyeri inflamasi sendi.")
            else:
                st.success("Risiko PsA Rendah saat ini. Lanjutkan monitoring.")
        
        st.caption("Diagnosis Potensial ini bersifat Skrining Awal (Modul Preventif).")


    with col_diag_2:
        st.subheader("2. Skrining Kulit AI (Upload Foto)")
        uploaded_file = st.file_uploader("Upload foto area kulit untuk dianalisis:", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Foto diunggah.', width=300)
            
            if st.button("Analisis Foto Kulit dengan Gemini Vision", type='primary', disabled=(client is None)):
                # Panggil Gemini Vision
                result_vision = analyze_skin_photo(uploaded_file)
                st.info("Hasil Analisis AI:")
                st.markdown(result_vision)
            
            if client is None: st.caption("Masukkan API Key untuk mengaktifkan Gemini Vision.")
        else:
            st.info("Unggah gambar kulit untuk cek potensi Psoriasis.")


# =========================================================================
# === TAB 3: INTERVENSI STRES & REHABILITASI
# =========================================================================

with tab_intervensi:
    st.header("Intervensi Cerdas Personal")

    col_int_1, col_int_2 = st.columns(2)

    with col_int_1:
        st.subheader("🧘 Mindfulness & Relaksasi Coach (Gemini)")
        if st.session_state.user_log:
            latest_data = st.session_state.user_log[-1]
            current_mood = "Frustrasi dan Tegang" if latest_data['stress_score'] > 7 else "Biasa Saja"
            hrv_status_text = "HRV sangat rendah, sinyal istirahat" if latest_data['hrv_avg'] < 0.4 else "HRV normal"
            
            if st.button("Mulai Sesi Stress Coaching Adaptif", type='primary', disabled=(client is None)):
                with st.spinner("Gemini menyiapkan panduan..."):
                    coaching_result = generate_stress_coaching_gemini(current_mood, hrv_status_text, datetime.now().strftime("%H:%M"))
                    st.markdown(f"**Hasil Coaching:**\n{coaching_result}")
            if client is None: st.caption("Masukkan API Key untuk mengaktifkan fitur ini.")

    with col_int_2:
        st.subheader("🤸 Tele-Rehabilitasi & Pacing Coach")
        if st.session_state.user_log:
            latest_data = st.session_state.user_log[-1]

            # Activity Pacing Coach (Gemini/ML Logic)
            if latest_data['hrv_avg'] < 0.5 or latest_data['stress_score'] > 7:
                pacing_advice = "🚨 **Pacing Coach:** Kelelahan/Stres tinggi terdeteksi. Kurangi intensitas latihan, prioritaskan istirahat."
                st.error(pacing_advice)
            else:
                pacing_advice = "**Pacing Coach:** Energi stabil. Pertahankan jadwal latihan."
                st.success(pacing_advice)
            
            # Computer Vision Feedback (Simulasi)
            if latest_data['video_rehab_status'] == 'Poor_Posture':
                st.error(f"**Vision Feedback:** Postur latihan buruk terdeteksi. Perlu koreksi segera!")
            elif latest_data['video_rehab_status'] == 'Fatigue':
                st.warning(f"**Vision Feedback:** Deteksi kelelahan saat latihan. Harap berhenti.")
            else:
                st.info("Feedback Vision: Latihan diselesaikan dengan baik.")

# =========================================================================
# === TAB 4: DASHBOARD KLINIS & EHR (Modul Integrasi Sistem)
# =========================================================================

with tab_dashboard:
    st.header("Dashboard Klinis, Riwayat, & Integrasi EHR")
    
    col_dash_1, col_dash_2 = st.columns([2, 1])

    with col_dash_1:
        st.subheader("Tren Aktivitas Penyakit")
        if not df_log.empty:
            df_log_viz = df_log.set_index('timestamp')[['pain_score', 'stress_score', 'hrv_avg']]
            st.line_chart(df_log_viz)
            st.caption("Grafik menunjukkan Nyeri (biru) vs. HRV (hijau).")
        
        st.subheader("Prediksi Terapi Biologics (Simulasi ML)")
        if st.button("Prediksi Respons Terapi Biologics", key='predict_drug'):
            drug_a_prob = random.uniform(0.65, 0.95)
            drug_b_prob = random.uniform(0.5, 0.8)
            st.code(
                f"Probabilitas Remisi Drug A (TNF Inhibitor): {drug_a_prob:.2f}\n"
                f"Probabilitas Remisi Drug B (IL-17 Inhibitor): {drug_b_prob:.2f}\n"
                "\n**Rekomendasi ML:** Drug A direkomendasikan untuk pasien ini (Cost-Effective)."
            )

    with col_dash_2:
        st.subheader("Riwayat Kesehatan Pasien")
        history_data = [
            {"Kategori": "Diagnosis Primer", "Detail": st.session_state.user_history['diagnosis']},
            {"Kategori": "Komorbiditas", "Detail": st.session_state.user_history['komorbiditas']},
            {"Kategori": "Obat Saat Ini", "Detail": st.session_state.user_history['obat_saat_ini']},
            {"Kategori": "Jumlah Log", "Detail": len(st.session_state.user_log)}
        ]
        st.table(pd.DataFrame(history_data))

        st.subheader("Ringkasan Klinis AI (EHR)")
        if st.session_state.user_log:
            if st.button("Generate AI Summary (Gemini)", key='generate_summary', type='primary', disabled=(client is None)):
                ai_summary = generate_clinician_summary(st.session_state.user_log, st.session_state.user_history)
                st.info("Ringkasan Klinis:")
                st.markdown(ai_summary)
            if client is None: st.caption("Fitur AI dinonaktifkan tanpa API Key.")
        
        st.caption("Data log mentah yang akan dikirim ke EHR:")
        st.dataframe(df_log[['timestamp', 'pain_score', 'hrv_avg', 'med_adherence']].tail(5), use_container_width=True)

# =========================================================================
# === TAB 5: ASISTEN KESEHATAN PsA (Character Bot)
# =========================================================================

with tab_chatbot:
    st.header("💬 Asisten Kesehatan PsA (Chatbot)")
    st.caption("Tanya jawab 24/7 seputar PsA, gaya hidup, dan efek samping umum.")

    # Tampilkan riwayat pesan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input pengguna (hanya aktif jika client tersedia)
    if prompt := st.chat_input("Tanyakan sesuatu tentang PsA...", disabled=(client is None)):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_response = generate_chatbot_response(prompt)
            st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if client is None: st.error("Chatbot dinonaktifkan. Harap masukkan API Key.")
```
