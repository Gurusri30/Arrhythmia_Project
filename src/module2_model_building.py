import os
import wfdb
import numpy as np
import scipy.signal as signal
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Flatten

# ==========================================
# 1. Base Paper Mapping: 5 AAMI Classes
# ==========================================
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}

def load_and_preprocess_multi_records(records, data_dir='data', window_sec=0.6):
    """
    Downloads and extracts heartbeats mapped to 5 AAMI classes from multiple records.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    all_beats = []
    all_labels = []
    
    for record_name in records:
        print(f"Processing Record {record_name}...")
        try:
            # Download if not exists
            if not os.path.exists(os.path.join(data_dir, record_name + '.dat')):
                wfdb.dl_database('mitdb', data_dir, records=[record_name])
            
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
            
            raw_signal = record.p_signal[:, 0]
            fs = record.fs
            
            # Denoise (Butterworth filter from Module 1)
            nyquist = 0.5 * fs
            b, a = signal.butter(3, [0.5/nyquist, 45.0/nyquist], btype='band')
            clean_signal = signal.filtfilt(b, a, raw_signal)
            
            # Segment
            window_samples = int(window_sec * fs)
            
            for idx, r_peak in enumerate(annotation.sample):
                symbol = annotation.symbol[idx]
                if symbol in aami_mapping:
                    start = r_peak - (window_samples // 2)
                    end = r_peak + (window_samples // 2)
                    
                    if start >= 0 and end < len(clean_signal):
                        beat = clean_signal[start:end]
                        # Min-Max Normalization
                        min_val, max_val = np.min(beat), np.max(beat)
                        if max_val != min_val:
                            beat = (beat - min_val) / (max_val - min_val)
                        
                        all_beats.append(beat)
                        all_labels.append(aami_mapping[symbol])
        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    return np.array(all_beats), np.array(all_labels)

# ==========================================
# 2. Enhanced Model: CNN + BiLSTM + Attention
# ==========================================
def build_enhanced_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Layer for spatial morphological features
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # BiLSTM Layer for temporal sequence dependencies
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    # Attention Mechanism (Focusing on abnormal segments)
    attention_out = Attention()([x, x])
    
    # Pooling and Classification
    x = GlobalAveragePooling1D()(attention_out)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x) # 5 Classes Output
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    print("\n--- STARTING MODULE 2: DATA BALANCING & ENHANCED AI MODELING ---")
    
    # We use a mix of records known to contain various arrhythmias (S, V, F, Q) to ensure multi-class availability
    selected_records = ['100', '119', '201', '203', '213']
    
    X, y = load_and_preprocess_multi_records(selected_records)
    print(f"\nExtracted Total Beats: {X.shape[0]}")
    print(f"Original Class Distribution: {np.bincount(y)}")
    
    # Step 2: SMOTE Data Balancing (From Base Paper)
    print("\n[Base Paper Integration] Applying SMOTE for Data Balancing...")
    # Flatten X for SMOTE
    X_flat = X.reshape(X.shape[0], X.shape[1])
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    # Reshape back to 3D for Deep Learning (samples, timesteps, features)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)
    print(f"Balanced Class Distribution: {np.bincount(y_resampled)}")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    print(f"\nTraining Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    # Step 3: Build Enhancement Architecture
    print("\n[Enhancement] Building CNN + BiLSTM + Attention Model...")
    model = build_enhanced_model((X_train.shape[1], 1))
    model.summary()
    
    # Step 4: Training the Model
    print("\nTraining the model... (This will take a moment)")
    # Using 5 epochs for quick demonstration (can be increased to 30-50 for final thesis results)
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
    
    # Step 5: Evaluation
    print("\nEvaluating the Model on Test Data...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"✅ MODEL TESTING ACCURACY: {acc * 100:.2f}%")
    print(f"==========================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['N', 'S', 'V', 'F', 'Q']))
    
    # Save the Enhanced Model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(f'{model_dir}/enhanced_arrhythmia_model.h5')
    print(f"\nModel successfully saved to {model_dir}/enhanced_arrhythmia_model.h5")
