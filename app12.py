import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import soundfile as sf

# Configuration
st.set_page_config(page_title="Drunk Detection", layout="wide")

# Load models
try:
    model = joblib.load("model_drunk.pkl")
    le = joblib.load("label_encoder_drunk.pkl")
except FileNotFoundError as e:
    st.error(f"Model Error: {str(e)}")
    st.stop()

# Session state initialization
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
            st.warning("Audio too short (min 1sec required)")
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
        st.error(f"Feature Error: {str(e)}")
        return None

def show_prediction(features):
    try:
        probs = model.predict_proba(features)[0]
        pred_idx = np.argmax(probs)
        label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100
        st.success(f"Result: {label.upper()} ({confidence:.1f}% confidence)")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

# WebRTC callback
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    array = frame.to_ndarray(format="f32le")
    st.session_state.audio_buffer.append(array)
    return frame

# Main app
st.title("🎤 Alcohol Impairment Detector")
input_method = st.sidebar.radio("Input Method", ("Upload File", "Live Recording"))

if input_method == "Upload File":
    uploaded = st.sidebar.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded.read())
            temp_path = f.name
        st.audio(temp_path)
        features = extract_features(temp_path)
        if features is not None:
            show_prediction(features)
        os.unlink(temp_path)

elif input_method == "Live Recording":
    st.sidebar.markdown("### Recording Instructions")
    st.sidebar.write("1. Click 'Start Recording'")
    st.sidebar.write("2. Speak for 5-10 seconds")
    st.sidebar.write("3. Click 'Stop'")
    
    ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
        ),
        audio_frame_callback=audio_frame_callback,
    )

    # Process after recording stops
    if ctx and not ctx.state.playing:
        if len(st.session_state.audio_buffer) > 0:
            try:
                audio = np.concatenate(st.session_state.audio_buffer)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, audio, 44100, subtype='PCM_16')
                    st.audio(f.name)
                    features = extract_features(f.name)
                    if features is not None:
                        show_prediction(features)
            finally:
                st.session_state.audio_buffer = []
                if os.path.exists(f.name):
                    os.unlink(f.name)

st.markdown("---")
st.caption("AI-powered voice analysis system")
