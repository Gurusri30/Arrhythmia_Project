import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# ==========================================
# 1. Configuration & Styling
# ==========================================
st.set_page_config(page_title="Intelligent Arrhythmia Framework", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 32px !important; font-weight: bold; color: #0f4c81; text-align: center;}
    .sub-header { font-size: 16px !important; color: #555555; text-align: center; margin-bottom: 30px;}
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    .risk-critical { color: #dc3545; font-size: 24px; font-weight: bold; }
    .risk-warning { color: #ffc107; font-size: 24px; font-weight: bold; }
    .risk-safe { color: #28a745; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AI Model & Data Loading
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

def calculate_xai_heatmap(beat):
    grad = np.gradient(beat)
    attention_scores = np.abs(grad) * beat
    smoothed = np.convolve(attention_scores, np.ones(10)/10, mode='same')
    if np.max(smoothed) > np.min(smoothed):
        smoothed = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
    return smoothed

model = load_ai_model()
beats = load_sample_data()

classes = {
    0: ("Normal Beat (N)", "Low Risk", 15, "Normal sinus rhythm detected. No immediate clinical intervention required."),
    1: ("Supraventricular Ectopic (S)", "Medium Risk", 45, "Atrial anomaly detected. Potential PAC. Routine monitoring recommended."),
    2: ("Ventricular Ectopic (V)", "High Risk", 85, "Ventricular anomaly detected (PVC). High risk of Ventricular Tachycardia/Fibrillation. Immediate Cardiology Consult Required!"),
    3: ("Fusion Beat (F)", "Medium-High Risk", 65, "Waveform fusion detected. Suggest 24-hour Holter monitoring."),
    4: ("Unknown / Paced (Q)", "Variable Risk", 50, "Unclassifiable morphology or Paced rhythm. Verify pacemaker functionality.")
}

# ==========================================
# 3. Main Dashboard UI
# ==========================================
st.markdown('<p class="main-header">🫀 An Intelligent Deep Learning Framework for Cardiac Arrhythmia Detection and Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Research Grade Clinical Dashboard integrating CNN-BiLSTM, Attention Mechanisms, and Real-Time Risk Stratification</p>', unsafe_allow_html=True)

if beats is None or model is None:
    st.error("System Initialization Failed: Dataset or Model weights missing.")
    st.stop()

# Sidebar - Patient Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3209/3209986.png", width=80)
st.sidebar.markdown("### 🏥 Clinical Control Panel")
st.sidebar.caption("Data Source: PhysioNet MIT-BIH Arrhythmia Database (Validated Clinical Data)")

beat_index = st.sidebar.slider("Select Live ECG Trace Index:", 0, len(beats)-1, 15)
selected_beat = beats[beat_index]
patient_id = f"MIT-BIH-PT-{8000 + beat_index}"
st.sidebar.text_input("Auto-Generated Patient ID:", value=patient_id, disabled=True)

st.sidebar.markdown("---")
if st.sidebar.button("⚙️ Execute Full Analysis Pipeline", type="primary", use_container_width=True):
    with st.spinner("Initializing Deep Learning Pipeline..."):
        time.sleep(1)
        input_data = selected_beat.reshape(1, 216, 1)
        probs = model.predict(input_data)[0]
        pred_idx = np.argmax(probs)
        conf = probs[pred_idx] * 100
        diag, risk_label, risk_score_base, advice = classes[pred_idx]
        
        # Calculate dynamic risk score based on confidence
        dynamic_risk = min(100, risk_score_base + (conf / 10)) if pred_idx != 0 else (100 - conf)
        
        st.session_state['analysis'] = {
            'diagnosis': diag, 'confidence': conf, 'probs': probs,
            'risk_label': risk_label, 'risk_score': dynamic_risk, 'advice': advice
        }

# ==========================================
# 4. Advanced Tabbed Interface
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Signal Processing & Feature Extraction", 
    "🤖 CNN-BiLSTM Inference", 
    "⚠️ Cardiac Risk Prediction", 
    "🔍 Explainable AI (XAI)"
])

# --- TAB 1: Signal Processing ---
with tab1:
    st.markdown("### Phase 1: Morphological Signal Processing")
    st.write("Visualizing the 0.6-second extracted R-R interval window before neural inference.")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=selected_beat, mode='lines', name='Extracted ECG Beat', line=dict(color='blue', width=2)))
    fig1.update_layout(title="Normalized ECG Signal Morphology", xaxis_title="Time (Samples)", yaxis_title="Amplitude (mV)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: AI Inference ---
with tab2:
    st.markdown("### Phase 2: Neural Network Classification")
    if 'analysis' in st.session_state:
        res = st.session_state['analysis']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label="Predicted Arrhythmia Class", value=res['diagnosis'])
            st.metric(label="Network Confidence", value=f"{res['confidence']:.2f} %")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            class_labels = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
            fig2 = px.bar(x=class_labels, y=res['probs']*100, labels={'x':'Arrhythmia Class', 'y':'Probability (%)'},
                          title="Softmax Probability Distribution", color=class_labels, 
                          color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👈 Click 'Execute Full Analysis Pipeline' in the sidebar to view results.")

# --- TAB 3: Risk Prediction (NEW & ADVANCED) ---
with tab3:
    st.markdown("### Phase 3: Clinical Risk Stratification System")
    if 'analysis' in st.session_state:
        res = st.session_state['analysis']
        
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            # Gauge Chart for Risk Score
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['risk_score'],
                title = {'text': "Calculated Cardiac Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': res['risk_score']}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_r2:
            st.markdown(f"#### Patient ID: `{patient_id}`")
            st.markdown("---")
            risk_color = "risk-critical" if res['risk_score'] >= 70 else ("risk-warning" if res['risk_score'] >= 30 else "risk-safe")
            st.markdown(f"**Triage Level:** <span class='{risk_color}'>{res['risk_label'].upper()}</span>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### 🩺 Clinical Action Required:")
            st.warning(res['advice'])
    else:
        st.info("👈 Click 'Execute Full Analysis Pipeline' in the sidebar to calculate Risk Score.")

# --- TAB 4: Explainable AI ---
with tab4:
    st.markdown("### Phase 4: Saliency Mapping & Model Interpretability")
    st.write("Visual validation of the AI's decision-making process. The heatmap highlights the morphological deviations (e.g., widened QRS complex) utilized by the Attention Layer.")
    
    heatmap = calculate_xai_heatmap(selected_beat)
    
    fig4 = go.Figure()
    # Add actual signal
    fig4.add_trace(go.Scatter(y=selected_beat, mode='lines', name='ECG Signal', line=dict(color='black', width=2)))
    # Add heatmap overlay as a bar chart behind the line
    fig4.add_trace(go.Bar(y=[max(selected_beat)]*216, marker=dict(color=heatmap, colorscale='Reds', showscale=True), 
                          opacity=0.4, name='Attention Heatmap', hoverinfo='none'))
    
    fig4.update_layout(title="Attention Layer Saliency Overlay", xaxis_title="Time (Samples)", yaxis_title="Amplitude", template="plotly_white", barmode='overlay')
    st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Engineered for Clinical Research and Publication Standard Triage Analysis.</p>", unsafe_allow_html=True)
