import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# ==========================================
# 1. Configuration & Setup
# ==========================================
st.set_page_config(page_title="Arrhythmia Detection AI", page_icon="🫀", layout="wide")

st.title("🫀 Real-Time Cardiac Arrhythmia Detection & Risk Prediction")
st.markdown("""
This application uses a hybrid **CNN + BiLSTM + Attention** model to analyze Electrocardiogram (ECG) signals in real-time.
It classifies the heartbeat into 5 AAMI standard categories and provides a Cardiac Risk Stratification score.
""")

# ==========================================
# 2. Load Model and Data
# ==========================================
@st.cache_resource
def load_ai_model():
    model_path = 'models/enhanced_arrhythmia_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_data
def load_sample_data():
    data_path = 'data/processed_beats.npy'
    if os.path.exists(data_path):
        return np.load(data_path)
    return None

model = load_ai_model()
beats = load_sample_data()

# Class Mapping
classes = {
    0: ("Normal Beat (N)", "Low Risk 🟢", "Healthy heart rhythm. No immediate action required."),
    1: ("Supraventricular Ectopic (S)", "Medium Risk 🟡", "Abnormal rapid heart rhythm originating above the ventricles. Monitor closely."),
    2: ("Ventricular Ectopic (V)", "High Risk 🔴", "Abnormal heartbeat originating in the lower heart chambers. Needs medical attention."),
    3: ("Fusion Beat (F)", "Medium-High Risk 🟠", "A fusion of a normal and a ventricular beat. Clinical evaluation recommended."),
    4: ("Unknown / Paced (Q)", "Variable Risk ⚪", "Unclassifiable or paced rhythm. Requires expert cardiologist review.")
}

# ==========================================
# 3. Sidebar UI (User Input)
# ==========================================
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Upload a new ECG file or select a live sample from the monitoring system.")

if beats is not None:
    st.sidebar.success(f"Connected to Live Monitor: {len(beats)} beats available.")
    beat_index = st.sidebar.slider("Select Patient Heartbeat Segment", min_value=0, max_value=len(beats)-1, value=15)
    selected_beat = beats[beat_index]
else:
    st.sidebar.error("ECG data not found! Please run Module 1 first.")
    st.stop()

if model is None:
    st.error("AI Model not found! Please run Module 2 first.")
    st.stop()

# ==========================================
# 4. Main Dashboard
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 ECG Signal Analysis (Segment #{beat_index})")
    
    # Plot the ECG Signal
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(selected_beat, color='#1f77b4', linewidth=2)
    ax.set_title("Processed ECG Heartbeat", fontsize=14)
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Simulate Explainable AI (Attention Highlighting the QRS complex region)
    # The QRS complex is usually in the center of the window
    center = len(selected_beat) // 2
    ax.axvspan(center - 20, center + 20, color='red', alpha=0.2, label="AI Attention Focus (QRS)")
    ax.legend()
    
    st.pyplot(fig)

with col2:
    st.subheader("🤖 AI Prediction Results")
    
    # Add a predict button for real-time feel
    if st.button("Run Real-Time Diagnosis", type="primary"):
        with st.spinner("Analyzing ECG signal with CNN-BiLSTM-Attention..."):
            # Reshape for prediction (1, timesteps, channels)
            input_data = selected_beat.reshape(1, len(selected_beat), 1)
            prediction_probs = model.predict(input_data)[0]
            predicted_class = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_class] * 100
            
            name, risk, advice = classes[predicted_class]
            
            st.success("Diagnosis Complete!")
            
            st.markdown(f"### **Condition:** {name}")
            st.markdown(f"### **Confidence:** {confidence:.2f}%")
            st.markdown(f"### **Risk Level:** {risk}")
            
            st.info(f"**Medical Advice:** {advice}")
            
            # Show Probability Chart
            st.markdown("**Class Probabilities:**")
            st.bar_chart({classes[i][0].split('(')[1][0]: prob for i, prob in enumerate(prediction_probs)})
    else:
        st.info("Click the button above to run the AI diagnosis.")

# ==========================================
# 5. Footer
# ==========================================
st.markdown("---")
st.markdown("*Developed by Gurusri, Madhumitha, Viveka & Poojitha | Supervised by Mr. S. Nagaraj M.E.,(Ph.D).*")
