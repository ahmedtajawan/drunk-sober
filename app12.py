# drunk_detection_app.py (final working version)
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

# Configuration
st.set_page_config(page_title="Drunk Detection", layout="wide")

# Load models
try:
    model_drunk = joblib.load("model_drunk.pkl")
    le_drunk = joblib.load("label_encoder_drunk.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.stop()

# Audio buffer initialization
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

# Audio processing functions
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
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            [np.mean(librosa.feature.zero_crossing_rate(y))],
            [np.mean(librosa.feature.rms(y=y))],
            [np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))],
            [librosa.beat.beat_track(y=y, sr=sr)[0]]
        ])
        return features.reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def show_prediction(features):
    drunk_probs = model_drunk.predict_proba(features)[0]
    drunk_idx = np.argmax(drunk_probs)
    drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
    drunk_conf = drunk_probs[drunk_idx] * 100
    st.success(f"🧠 Prediction: {drunk_label.upper()} ({drunk_conf:.2f}% confidence)")

# WebRTC callback
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    array = frame.to_ndarray(format="f32le")
    st.session_state.audio_buffer.append(array)
    return frame

# Main app
st.title("🎤 Drunk/Sober Classifier")
option = st.sidebar.radio("Input Method", ("Upload Audio File", "Record Audio"))

if option == "Upload Audio File":
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
    st.sidebar.write("1. Click 'START RECORDING'")
    st.sidebar.write("2. Speak for 5-10 seconds")
    st.sidebar.write("3. Click 'STOP'")
    
    ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
        ),
        audio_frame_callback=audio_frame_callback,
    )

    # Corrected state handling
    if ctx and not ctx.state.playing and len(st.session_state.audio_buffer) > 0:
        try:
            audio_array = np.concatenate(st.session_state.audio_buffer)
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sf.write(temp_path, audio_array, 44100, subtype='PCM_16')
            
            st.audio(temp_path)
            features = extract_features(temp_path)
            if features is not None:
                show_prediction(features)
        finally:
            st.session_state.audio_buffer = []
            if os.path.exists(temp_path):
                os.unlink(temp_path)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
