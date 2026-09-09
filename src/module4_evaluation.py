import os
import wfdb
import numpy as np
import scipy.signal as signal
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D

# ==========================================
# 1. Setup & Definitions
# ==========================================
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}
class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def load_ai_model(model_path):
    print("Loading Enhanced Model Architecture...")
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

def get_test_data():
    print("Generating Test Dataset for Evaluation...")
    records = ['119', '201', '213']
    data_dir = 'data'
    all_beats, all_labels = [], []
    
    for record_name in records:
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
        raw_signal = record.p_signal[:, 0]
        fs = record.fs
        
        nyquist = 0.5 * fs
        b, a = signal.butter(3, [0.5/nyquist, 45.0/nyquist], btype='band')
        clean_signal = signal.filtfilt(b, a, raw_signal)
        
        window_samples = int(0.6 * fs)
        
        for idx, r_peak in enumerate(annotation.sample):
            symbol = annotation.symbol[idx]
            if symbol in aami_mapping:
                start = r_peak - (window_samples // 2)
                end = r_peak + (window_samples // 2)
                if start >= 0 and end < len(clean_signal):
                    beat = clean_signal[start:end]
                    min_val, max_val = np.min(beat), np.max(beat)
                    if max_val != min_val:
                        beat = (beat - min_val) / (max_val - min_val)
                    all_beats.append(beat)
                    all_labels.append(aami_mapping[symbol])
                    
    X = np.array(all_beats)
    y = np.array(all_labels)
    
    X_flat = X.reshape(X.shape[0], X.shape[1])
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    return X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1), y_resampled

if __name__ == "__main__":
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    model = load_ai_model('models/enhanced_arrhythmia_model.h5')
    X_test, y_test = get_test_data()
    
    print("Running Predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 1. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: CNN-BiLSTM-Attention', fontsize=16)
    plt.ylabel('True Arrhythmia Class', fontsize=12)
    plt.xlabel('Predicted Arrhythmia Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"Saved: {results_dir}/confusion_matrix.png")
    
    # 2. Save Classification Report as Text
    print("Generating Classification Report...")
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(os.path.join(results_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("=== FINAL MODULE 4 EVALUATION REPORT ===\n\n")
        f.write("Model: Enhanced CNN + BiLSTM + Attention\n")
        f.write("Dataset: MIT-BIH Arrhythmia Database (Balanced via SMOTE)\n\n")
        f.write(report)
    print(f"Saved: {results_dir}/evaluation_report.txt")
    
    print("\n[SUCCESS] Module 4 Evaluation complete! Graphs are saved in the 'results' folder.")
