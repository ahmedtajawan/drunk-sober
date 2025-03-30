import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
import av
import soundfile as sf
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import threading
from typing import List
import time

# Suppress unnecessary warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*missing ScriptRunContext.*")

# =====================
# 🎨 Streamlit Configuration
# =====================
st.set_page_config(page_title="Drunk Detection", layout="wide")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'processor' not in st.session_state:
    st.session_state.processor = None


# =====================
# 🎯 Load Models and Encoders
# =====================
@st.cache_resource
def load_models():
    try:
        model_drunk = joblib.load("model_drunk.pkl")
        le_drunk = joblib.load("label_encoder_drunk.pkl")
        return model_drunk, le_drunk
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


model_drunk, le_drunk = load_models()


# =====================
# 🔍 Feature Extraction
# =====================
def extract_features(y, sr):
    try:
        if len(y) < sr:
            st.warning("⚠️ Audio is too short for feature extraction (needs at least 1 second).")
            return None

        # Extract Features
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


# =====================
# 🎙️ Live Recording Handling
# =====================
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # Reduced sample rate for better compatibility
        self.channels = 1
        self._frames = []
        self._lock = threading.Lock()

    def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        if st.session_state.recording:
            with self._lock:
                for frame in frames:
                    audio_array = frame.to_ndarray().astype(np.float32)
                    self._frames.append(audio_array)
                    print(f"Captured {len(audio_array)} samples")  # Debug
        return frames

    def get_audio(self):
        with self._lock:
            if not self._frames:
                return None, None
            audio_array = np.concatenate(self._frames)
            self._frames.clear()
        print(f"Total samples captured: {len(audio_array)}")  # Debug
        return audio_array, self.sample_rate


# =====================
# 🧠 Show Prediction
# =====================
def show_prediction(features):
    try:
        drunk_probs = model_drunk.predict_proba(features)[0]
        drunk_idx = np.argmax(drunk_probs)
        drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
        drunk_conf = drunk_probs[drunk_idx] * 100

        st.success("🧠 Prediction Results")
        st.markdown(f"- **Condition**: `{drunk_label.upper()}` ({drunk_conf:.2f}% confidence)")

        if drunk_label.lower() == "drunk":
            st.warning("⚠️ Warning: Potential intoxication detected")
        else:
            st.balloons()
    except Exception as e:
        st.error(f"Error making prediction: {e}")


# =====================
# 📤 File Upload Section
# =====================
st.title("🎤 Drunk/Sober Classifier")
st.sidebar.write("### 📂 Upload Audio File")
uploaded_file = st.sidebar.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner("Processing audio..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                if uploaded_file.name.endswith('.mp3'):
                    audio = AudioSegment.from_mp3(uploaded_file)
                    audio.export(tmp.name, format="wav")
                else:
                    tmp.write(uploaded_file.getvalue())

                y, sr = librosa.load(tmp.name, sr=None)
                os.unlink(tmp.name)

            st.audio(uploaded_file.getvalue())
            features = extract_features(y, sr)

            if features is not None:
                show_prediction(features)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# =====================
# 🎙️ Live Recording Section
# =====================
st.sidebar.write("### 🎙️ Record Live Audio")


def audio_processor_factory():
    processor = AudioRecorder()
    st.session_state.processor = processor
    return processor


webrtc_ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=audio_processor_factory,
    media_stream_constraints={
        "audio": {
            "sampleRate": 16000,
            "channelCount": 1,
            "autoGainControl": False,
            "echoCancellation": False,
            "noiseSuppression": False
        }
    },
    frontend_rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    server_rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True
)

# Recording controls
if webrtc_ctx.audio_processor:
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🎤 Start Recording"):
            st.session_state.recording = True
            st.toast("Recording started", icon="🎙️")

    with col2:
        if st.button("🛑 Stop Recording"):
            st.session_state.recording = False
            time.sleep(1)  # Increased delay for final frames

            if st.session_state.processor:
                audio_array, sr = st.session_state.processor.get_audio()

                if audio_array is not None and len(audio_array) > 0:
                    with st.spinner("Analyzing recording..."):
                        try:
                            # Convert and normalize audio
                            audio_int16 = (audio_array * 32767).astype(np.int16)
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                sf.write(tmp.name, audio_int16, sr, subtype='PCM_16')
                                st.audio(tmp.name)

                                # Load audio using librosa
                                y, sr = librosa.load(tmp.name, sr=sr)
                                features = extract_features(y, sr)

                                if features is not None:
                                    show_prediction(features)
                                else:
                                    st.error("Feature extraction failed")

                                os.unlink(tmp.name)
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
                else:
                    st.warning("No audio recorded. Please try again.")

if st.session_state.recording:
    st.warning("🔴 Recording in progress... Speak now!")

# =====================
# Footer
# =====================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Librosa, and Scikit-learn")