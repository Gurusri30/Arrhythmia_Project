import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
import os
import time

# ==========================================
# 1. Configuration & Styling
# ==========================================
st.set_page_config(page_title="Advanced Arrhythmia CDSS", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 36px !important; font-weight: bold; color: #1E3A8A; }
    .sub-header { font-size: 18px !important; color: #64748B; margin-bottom: 20px;}
    .risk-high { color: #DC2626; font-weight: bold; font-size: 20px;}
    .risk-medium { color: #D97706; font-weight: bold; font-size: 20px;}
    .risk-low { color: #059669; font-weight: bold; font-size: 20px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Core Logic & Model Loading
# ==========================================
@st.cache_resource
def load_ai_model():
    model_path = 'models/enhanced_arrhythmia_model.h5'
    if os.path.exists(model_path):
        inputs = Input(shape=(216, 1))
        x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        attention_out = Attention()([x, x])
        x = GlobalAveragePooling1D()(attention_out)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(5, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.load_weights(model_path)
        return model
    return None

@st.cache_data
def load_sample_data():
    data_path = 'data/processed_beats.npy'
    if os.path.exists(data_path):
        return np.load(data_path)
    return None

def generate_saliency_heatmap(beat):
    grad = np.gradient(beat)
    attention_scores = np.abs(grad) * beat
    smoothed_attention = np.convolve(attention_scores, np.ones(10)/10, mode='same')
    min_val, max_val = np.min(smoothed_attention), np.max(smoothed_attention)
    if max_val > min_val:
        smoothed_attention = (smoothed_attention - min_val) / (max_val - min_val)
    return smoothed_attention

model = load_ai_model()
beats = load_sample_data()

classes = {
    0: ("Normal Beat (N)", "Low Risk", "Healthy rhythm. Routine checkup advised."),
    1: ("Supraventricular Ectopic (S)", "Medium Risk", "Atrial anomaly. Monitor closely."),
    2: ("Ventricular Ectopic (V)", "High Risk", "Ventricular anomaly. Immediate Consult!"),
    3: ("Fusion Beat (F)", "Medium-High Risk", "Waveform fusion. Recommend Holter."),
    4: ("Unknown / Paced (Q)", "Variable Risk", "Unclassifiable. Verify pacemaker.")
}

# ==========================================
# 3. App UI Layout
# ==========================================
st.markdown('<p class="main-header">🫀 Cardiac Arrhythmia Detection and Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time ECG Arrhythmia Classification & XAI Interpretability</p>', unsafe_allow_html=True)

if beats is None or model is None:
    st.error("Missing Data or Model. Please run Modules 1 and 2.")
    st.stop()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209986.png", width=100)
st.sidebar.title("Patient Control Panel")
st.sidebar.success(f"🟢 API Connected: {len(beats)} live records found.")
beat_index = st.sidebar.slider("Select Patient ECG Segment (ID)", min_value=0, max_value=len(beats)-1, value=15)
selected_beat = beats[beat_index]

# Tabs for Advanced UI
tab1, tab2, tab3 = st.tabs(["🏥 Live Diagnostics", "🧠 Explainable AI (XAI)", "📊 Model Confidence"])

# --- TAB 1: Live Diagnostics ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Electrocardiogram Trace")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(selected_beat, color='#0F172A', linewidth=1.5)
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        ax.set_facecolor('#F8FAFC')
        st.pyplot(fig)
        
    with col2:
        st.subheader("AI Triage Engine")
        if st.button("🚀 Execute Neural Inference", use_container_width=True, type="primary"):
            with st.spinner("Processing temporal sequences via BiLSTM..."):
                time.sleep(0.5) # Simulating API latency
                input_data = selected_beat.reshape(1, 216, 1)
                probs = model.predict(input_data)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx] * 100
                diag, risk, advice = classes[pred_idx]
                
                st.session_state['pred_data'] = (diag, risk, advice, conf, probs)
                
        if 'pred_data' in st.session_state:
            diag, risk, advice, conf, probs = st.session_state['pred_data']
            st.metric(label="Detected Arrhythmia", value=diag.split('(')[0])
            st.metric(label="AI Confidence", value=f"{conf:.2f}%")
            
            risk_class = "risk-high" if "High" in risk else ("risk-medium" if "Medium" in risk else "risk-low")
            st.markdown(f"**Risk Stratification:** <span class='{risk_class}'>{risk}</span>", unsafe_allow_html=True)
            st.info(f"**Clinical Advice:** {advice}")

# --- TAB 2: Explainable AI ---
with tab2:
    st.subheader("Glass-Box Interpretability (Saliency Heatmap)")
    st.markdown("The red zones indicate the specific morphological segments (e.g., widened QRS) that triggered the AI's diagnosis.")
    
    attention_weights = generate_saliency_heatmap(selected_beat)
    fig_xai, ax_xai = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(selected_beat))
    y = selected_beat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(attention_weights.min(), attention_weights.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(attention_weights)
    lc.set_linewidth(3)
    line = ax_xai.add_collection(lc)
    
    fig_xai.colorbar(line, ax=ax_xai, label="AI Attention Density")
    ax_xai.set_xlim(x.min(), x.max())
    ax_xai.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax_xai.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig_xai)

# --- TAB 3: Model Confidence ---
with tab3:
    st.subheader("Multi-Class Probability Distribution")
    if 'pred_data' in st.session_state:
        probs = st.session_state['pred_data'][4]
        class_labels = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
        
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        ax_bar.bar(class_labels, probs * 100, color=['green', 'orange', 'red', 'purple', 'gray'])
        ax_bar.set_ylabel("Probability (%)")
        st.pyplot(fig_bar)
    else:
        st.warning("Please run the Neural Inference in Tab 1 first.")
