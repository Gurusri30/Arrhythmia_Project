from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
import os

# ==========================================
# Module 6: IoMT API Integration (FastAPI)
# ==========================================

app = FastAPI(
    title="Arrhythmia IoMT Cloud API",
    description="REST API for IoT sensors and wearable devices to send real-time ECG data for Arrhythmia diagnosis.",
    version="1.0.0"
)

# Load Model Configuration
model = None
classes = {
    0: ("Normal Beat (N)", "Low Risk"),
    1: ("Supraventricular Ectopic (S)", "Medium Risk"),
    2: ("Ventricular Ectopic (V)", "High Risk"),
    3: ("Fusion Beat (F)", "Medium-High Risk"),
    4: ("Unknown / Paced (Q)", "Variable Risk")
}

@app.on_event("startup")
async def startup_event():
    global model
    model_path = '../models/enhanced_arrhythmia_model.h5'
    if os.path.exists(model_path):
        # Build architecture
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
        print("IoMT API Model Loaded Successfully.")
    else:
        print("Model not found. Run module 2.")

# Input Schema for IoT Devices
class ECGData(BaseModel):
    patient_id: str
    ecg_signal: list[float] # Expecting 216 float values

@app.get("/")
def read_root():
    return {"message": "Welcome to the Arrhythmia IoMT Cloud API. Endpoint active."}

@app.post("/predict")
def predict_arrhythmia(data: ECGData):
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model not loaded.")
        
    if len(data.ecg_signal) != 216:
        raise HTTPException(status_code=400, detail="ECG signal must contain exactly 216 data points (0.6s window).")
        
    try:
        # Convert to numpy array and reshape for model (1, 216, 1)
        signal_array = np.array(data.ecg_signal).reshape(1, 216, 1)
        
        # Inference
        predictions = model.predict(signal_array)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class])
        
        diagnosis, risk = classes[predicted_class]
        
        return {
            "patient_id": data.patient_id,
            "status": "SUCCESS",
            "diagnosis": diagnosis,
            "risk_level": risk,
            "confidence_score": round(confidence * 100, 2),
            "device_instruction": "Trigger Alert" if "High" in risk else "Normal Monitoring"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This block allows running via `python src/module6_api.py` locally for testing
    print("[INFO] Starting Module 6 IoMT API Server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
