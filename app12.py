import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
from pydub import AudioSegment
import wavio
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

audio_data = None  # To hold the recorded audio

audio_frames = []

def audio_frame_callback(frame):
    global audio_frames
    # Append each frame’s data to the list
    audio_frames.append(frame.to_ndarray(format="s16le"))
    return frame
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
# 📁 Audio Loading Function
# =====================
def load_audio(file_path):
    if file_path.endswith(".mp3"):
        audio = AudioSegment.from_file(file_path, format="mp3")
        audio.export("temp.wav", format="wav")
        file_path = "temp.wav"

    y, sr = librosa.load(file_path, sr=None)
    return y, sr


# =====================
# 🔍 Feature Extraction
# =====================
def extract_features(path):
    try:
        y, sr = load_audio(path)
        y, _ = librosa.effects.trim(y)

        if len(y) < sr:
            st.warning("⚠️ Audio is too short for feature extraction.")
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
# 🧠 Show Prediction
# =====================
def show_prediction(features):
    drunk_probs = model_drunk.predict_proba(features)[0]
    drunk_idx = np.argmax(drunk_probs)
    drunk_label = le_drunk.inverse_transform([drunk_idx])[0]
    drunk_conf = drunk_probs[drunk_idx] * 100

    st.success("🧠 Prediction Results")
    st.markdown(f"- **Condition**: `{drunk_label.upper()}` ({drunk_conf:.2f}% confidence)")




# =====================
# 📤 File Upload Section
# =====================
st.title("🎤 Drunk/Sober Classifier")

# Sidebar for Uploading or Recording
option = st.sidebar.radio("Choose Audio Input Method", ("Upload Audio File", "Record Audio"))

temp_path = None

if option == "Upload Audio File":
    st.sidebar.write("### 📂 Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Upload a .wav or .mp3 audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        st.audio(temp_path)
        features = extract_features(temp_path)
        if features is not None:
            show_prediction(features)

elif option == "Record Audio":
    st.write("### 🎙️ Record Your Audio")

    # Start the WebRTC streamer to record audio frames
    webrtc_ctx = webrtc_streamer(
        key="recording",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback,
    )
    
    # Check if any audio frames were captured (you might want to provide a button to finalize recording)
    if len(audio_frames) > 0:
        # Concatenate all audio frames along the first axis
        audio_data = np.concatenate(audio_frames, axis=0)
        
        # Write the combined audio to a temporary .wav file using soundfile
        import soundfile as sf
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(temp_path, audio_data, 44100, subtype='PCM_16')
        
        st.audio(temp_path, format="audio/wav")
        
        # Process the recorded audio
        features = extract_features(temp_path)
        if features is not None:
            show_prediction(features)


# =====================
# Footer
# =====================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Librosa, and Scikit-learn")
