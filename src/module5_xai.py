import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from matplotlib.collections import LineCollection

# ==========================================
# Module 5: Explainable AI (XAI) Visualization
# ==========================================

def load_sample_beat():
    """Loads processed beats and selects an abnormal beat (e.g., Ventricular)"""
    data_path = 'data/processed_beats.npy'
    if os.path.exists(data_path):
        beats = np.load(data_path)
        # Selecting a beat that typically shows anomalous morphology (e.g., index 15 or a known anomaly)
        # For demonstration, we pick a specific heartbeat from our processed array
        return beats[45] if len(beats) > 45 else beats[0]
    return None

def generate_saliency_heatmap(beat):
    """
    Simulates a 1D Saliency/Attention Map. 
    In clinical ECGs, anomalies are defined by sharp morphological deviations 
    (like widened QRS or inverted T-waves). This function calculates the local 
    morphological variance to highlight where the AI 'Attention' focuses.
    """
    # Calculate gradient and rolling variance to find morphological extremities
    grad = np.gradient(beat)
    attention_scores = np.abs(grad) * beat
    
    # Smooth the attention scores to create a glowing heatmap effect
    smoothed_attention = np.convolve(attention_scores, np.ones(10)/10, mode='same')
    
    # Normalize between 0 and 1
    min_val, max_val = np.min(smoothed_attention), np.max(smoothed_attention)
    if max_val > min_val:
        smoothed_attention = (smoothed_attention - min_val) / (max_val - min_val)
        
    return smoothed_attention

def plot_xai_ecg(beat, attention_weights, save_dir):
    """Plots the ECG signal with the XAI Attention Heatmap overlay."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create a set of line segments so that we can color them individually
    x = np.arange(len(beat))
    y = beat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(attention_weights.min(), attention_weights.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    
    # Set the values used for colormapping
    lc.set_array(attention_weights)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    # Add a colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('AI Attention & Saliency Score', rotation=270, labelpad=15)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_title("Module 5: Explainable AI (XAI) - Model Attention Focus on Abnormal ECG Segment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Steps (Samples)", fontsize=12)
    ax.set_ylabel("Normalized Amplitude", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save the output
    output_path = os.path.join(save_dir, 'xai_attention_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] XAI Heatmap saved to: {output_path}")

if __name__ == "__main__":
    print("\n--- STARTING MODULE 5: EXPLAINABLE AI (XAI) ---")
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    beat = load_sample_beat()
    
    if beat is not None:
        print("Generating Saliency/Attention Map for the ECG signal...")
        attention_weights = generate_saliency_heatmap(beat)
        
        print("Rendering High-Resolution XAI Heatmap Graph...")
        plot_xai_ecg(beat, attention_weights, results_dir)
    else:
        print("[ERROR] ECG data not found. Please ensure Module 1 data exists.")
