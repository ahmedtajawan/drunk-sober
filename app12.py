import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import soundfile as sf

# Initialize session state variables
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
if 'recording_finished' not in st.session_state:
    st.session_state.recording_finished = False
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    if not st.session_state.recording_finished:
        # Convert audio frame to numpy array
        frame_data = frame.to_ndarray(format="s16le")
        # Record sample rate from the first frame
        if st.session_state.sample_rate is None:
            st.session_state.sample_rate = frame.sample_rate
        st.session_state.audio_frames.append(frame_data)
    return frame

# 🎨 Streamlit Configuration
st.set_page_config(page_title="Drunk Detection", layout="wide")

# 🎯 Load Models and Encoders
try:
    model_drunk = joblib.load("model_drunk.pkl")
    le_drunk = joblib.load("label_encoder_drunk.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Model or label encoder file not found:\n{e}")
    st.stop()

# 📁 Audio Loading Function (same as before)
def load_audio(file_path):
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_file(file_path, format="mp3")
        audio.export("temp.wav", format="wav")
        file_path = "temp.wav"
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# 🔍 Feature Extraction (same as before)
COLUMN_NAMES = [f"mfcc_mean_{i}" for i in range(13)] + \
               [f"mfcc_std_{i}" for i in range(13)] + \
               ["zcr", "rms", "centroid", "tempo"]

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
        return pd.DataFrame([features], columns=COLUMN_NAMES)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# 🧠 Show Prediction (same as before)
def show_prediction(features):
    drunk_probs = model_drunk.predict_proba(features)[0]
    drunk_idx = np.argmax(drunk_probs)
    drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
    drunk_conf = drunk_probs[drunk_idx] * 100
    st.success("🧠 Prediction Results")
    st.markdown(f"- **Condition**: `{drunk_label.upper()}` ({drunk_conf:.2f}% confidence)")

# 🎤 Main App
st.title("🎤 Drunk/Sober Classifier")
option = st.sidebar.radio("Choose Audio Input Method", ("Upload Audio File", "Record Audio"))

if option == "Upload Audio File":
    # ... (same upload handling as before) ...

elif option == "Record Audio":
    st.write("### 🎙️ Record Your Audio")
    
    # Reset recording state when starting a new recording
    st.session_state.recording_finished = False
    st.session_state.audio_frames = []
    st.session_state.sample_rate = None

    webrtc_ctx = webrtc_streamer(
        key="recording",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback,
    )

    if st.button("Stop and Process Recording"):
        st.session_state.recording_finished = True
        if len(st.session_state.audio_frames) == 0:
            st.warning("No audio recorded. Please record first.")
        else:
            # Combine frames and save
            audio_data = np.concatenate(st.session_state.audio_frames, axis=0)
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sf.write(
                temp_path,
                audio_data,
                st.session_state.sample_rate,
                subtype="PCM_16"
            )
            st.audio(temp_path, format="audio/wav")
            
            # Process audio
            features = extract_features(temp_path)
            if features is not None:
                show_prediction(features)

            # Cleanup
            st.session_state.audio_frames = []
            st.session_state.sample_rate = None
