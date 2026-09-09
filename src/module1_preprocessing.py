import os
import wfdb
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# Member 1: Data Collection & Loading
# ==========================================
def download_and_load_data(record_name='100', data_dir='data'):
    """
    Downloads a specific record from MIT-BIH dataset and loads the signal.
    """
    print(f"--- Member 1 Task: Downloading Record {record_name} ---")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download the record (only if not already downloaded)
    try:
        wfdb.dl_database('mitdb', data_dir, records=[record_name])
    except Exception as e:
        print("Download issue (might already exist):", e)
        
    # Read the record and annotation
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Extract the first channel (usually MLII - modified limb lead II)
    raw_signal = record.p_signal[:, 0]
    r_peaks = annotation.sample
    
    print("Data loaded successfully!")
    return raw_signal, r_peaks, record.fs

# ==========================================
# Member 2: Noise Removal (Denoising)
# ==========================================
def denoise_signal(raw_signal, fs):
    """
    Applies a Butterworth Bandpass filter to remove baseline wander and high-frequency noise.
    """
    print("--- Member 2 Task: Denoising Signal ---")
    # Bandpass filter between 0.5 Hz and 45 Hz
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 45.0 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    
    clean_signal = signal.filtfilt(b, a, raw_signal)
    print("Signal denoised successfully!")
    return clean_signal

# ==========================================
# Member 3: Heartbeat Segmentation
# ==========================================
def segment_beats(clean_signal, r_peaks, fs, window_sec=0.6):
    """
    Segments the continuous signal into individual heartbeats centered around R-peaks.
    """
    print("--- Member 3 Task: Segmenting Heartbeats ---")
    window_samples = int(window_sec * fs) # Points to take before and after R-peak
    beats = []
    
    for r in r_peaks:
        start = r - (window_samples // 2)
        end = r + (window_samples // 2)
        
        # Ensure we don't go out of bounds
        if start >= 0 and end < len(clean_signal):
            beat = clean_signal[start:end]
            beats.append(beat)
            
    print(f"Extracted {len(beats)} individual heartbeats!")
    return np.array(beats)

# ==========================================
# Member 4: Normalization & Visualization
# ==========================================
def normalize_and_save(beats, raw_signal, clean_signal, r_peaks, fs, save_path='data/processed_beats.npy'):
    """
    Normalizes the beats using Min-Max scaling, plots the results, and saves the array.
    """
    print("--- Member 4 Task: Normalization and Saving ---")
    
    # Min-Max Normalization (Scale between 0 and 1)
    normalized_beats = []
    for beat in beats:
        min_val = np.min(beat)
        max_val = np.max(beat)
        if max_val != min_val:
            norm_beat = (beat - min_val) / (max_val - min_val)
        else:
            norm_beat = beat
        normalized_beats.append(norm_beat)
    
    normalized_beats = np.array(normalized_beats)
    
    # Save to file
    np.save(save_path, normalized_beats)
    print(f"Saved {len(normalized_beats)} normalized beats to {save_path}")
    
    # Plotting to Verify
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw vs Clean Signal (first 1000 samples)
    plt.subplot(3, 1, 1)
    plt.plot(raw_signal[:1000], label='Raw Signal', color='lightgray')
    plt.plot(clean_signal[:1000], label='Clean Signal', color='blue')
    
    # Plot R-peaks on the clean signal
    r_peaks_plot = [r for r in r_peaks if r < 1000]
    plt.plot(r_peaks_plot, clean_signal[r_peaks_plot], 'ro', label='R-Peaks')
    
    plt.title("Original vs Denoised ECG Signal")
    plt.legend()
    
    # Plot 2: A single extracted heartbeat before normalization
    plt.subplot(3, 1, 2)
    plt.plot(beats[0], color='orange')
    plt.title("Single Extracted Heartbeat (Before Normalization)")
    
    # Plot 3: A single extracted heartbeat after normalization
    plt.subplot(3, 1, 3)
    plt.plot(normalized_beats[0], color='green')
    plt.title("Single Extracted Heartbeat (Normalized 0 to 1)")
    
    plt.tight_layout()
    plot_path = 'data/module1_output_graph.png'
    plt.savefig(plot_path)
    print(f"Graph saved as {plot_path}")
    print("\n[SUCCESS] MODULE 1 COMPLETED SUCCESSFULLY!")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Starting Module 1: Data Collection & Preprocessing...")
    
    # Step 1
    raw_sig, r_peaks, fs = download_and_load_data(record_name='100')
    
    # Step 2
    clean_sig = denoise_signal(raw_sig, fs)
    
    # Step 3
    segmented_beats = segment_beats(clean_sig, r_peaks, fs)
    
    # Step 4
    normalize_and_save(segmented_beats, raw_sig, clean_sig, r_peaks, fs)
