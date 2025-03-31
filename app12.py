# drunk_detection_app.py
import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import soundfile as sf
import os

# =====================
# 🎨 Streamlit Configuration
# =====================
st.set_page_config(page_title="Drunk Detection", layout="wide")

# =====================
# 🎯 Load Models and Encoders
# =====================
try:
    model_drunk = joblib.load("model_drunk.pkl")
    le_drunk = joblib.load("label_encoder_drunk.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Model or label encoder file not found:\n{e}")
    st.stop()

# =====================
# 🔄 Audio Buffer Setup
# =====================
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

# =====================
# 📁 Audio Processing
# =====================
def load_audio(file_path):
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_file(file_path, format="mp3")
        audio.export("temp.wav", format="wav")
        file_path = "temp.wav"
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(path):
    try:
        y, sr = load_audio(path)
        y, _ = librosa.effects.trim(y)

        if len(y) < sr:
            st.warning("⚠️ Audio is too short for feature extraction.")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])

        features = mfcc_mean + mfcc_std + [zcr, rms, centroid, tempo]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def show_prediction(features):
    drunk_probs = model_drunk.predict_proba(features)[0]
    drunk_idx = np.argmax(drunk_probs)
    drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
    drunk_conf = drunk_probs[drunk_idx] * 100
    st.success("🧠 Prediction Results")
    st.markdown(f"- **Condition**: `{drunk_label.upper()}` ({drunk_conf:.2f}% confidence)")

# =====================
# 🎤 Audio Recording
# =====================
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    array = frame.to_ndarray(format="f32le")
    st.session_state.audio_buffer.append(array)
    return frame

# =====================
# 🖥️ Main Application
# =====================
st.title("🎤 Drunk/Sober Classifier")

option = st.sidebar.radio("Choose Input Method", ("Upload Audio File", "Record Audio"))

if option == "Upload Audio File":
    st.sidebar.header("📂 Upload Audio")
    uploaded_file = st.sidebar.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        st.audio(temp_path)
        features = extract_features(temp_path)
        if features is not None:
            show_prediction(features)

elif option == "Record Audio":
    st.sidebar.header("🎙️ Live Recording")
    st.sidebar.write("Click 'Start Recording' below and speak for 5-10 seconds")
    
    webrtc_ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
        ),
        audio_frame_callback=audio_frame_callback,
    )

    if webrtc_ctx and (webrtc_ctx.state.ice_connection_state == "closed" or not webrtc_ctx.state.playing):
        if len(st.session_state.audio_buffer) > 0:
            audio_array = np.concatenate(st.session_state.audio_buffer)
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Save as 16-bit PCM format for compatibility
            sf.write(temp_path, audio_array, 44100, subtype='PCM_16')
            
            st.audio(temp_path)
            features = extract_features(temp_path)
            if features is not None:
                show_prediction(features)
            
            # Cleanup
            st.session_state.audio_buffer = []
            os.unlink(temp_path)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Librosa, and Scikit-learn")
