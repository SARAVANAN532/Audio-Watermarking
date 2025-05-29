import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import os

# Function to add Gaussian noise
def add_noise(audio, noise_level=0.001):
    noise = np.random.normal(0, noise_level, audio.shape)
    noisy_audio = audio + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)  # Ensure valid range
    return noisy_audio

# Function to normalize audio to range [-1, 1]
def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

# Load WAV file
def load_audio(file_path):
    rate, data = wav.read(file_path)
    data = data.astype(np.float32)
    
    # Normalize if needed
    if data.dtype != np.float32 or np.max(np.abs(data)) > 1:
        data = normalize_audio(data)
    return rate, data

# Save WAV file
def save_audio(file_path, rate, data):
    # Rescale to int16 for saving
    data_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
    wav.write(file_path, rate, data_int16)

# Main processing function
def process_audio(input_file, output_dir):
    rate, data = load_audio(input_file)

    # If stereo, convert to mono for simplicity
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Add noise
    noisy_data = add_noise(data)

    # Denoise
    denoised_data = nr.reduce_noise(y=noisy_data, sr=rate)

    # Save all versions
    os.makedirs(output_dir, exist_ok=True)
    save_audio(os.path.join(output_dir, "original.wav"), rate, data)
    save_audio(os.path.join(output_dir, "noisy.wav"), rate, noisy_data)
    save_audio(os.path.join(output_dir, "denoised.wav"), rate, denoised_data)

    print("Files saved in:", output_dir)

# Example usage
input_audio_path = "MIDT.wav"
output_folder = "processed_audio"

process_audio(input_audio_path, output_folder)
