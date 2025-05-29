import numpy as np
import scipy.io.wavfile as wav
import os
import tensorflow as tf
from scipy.io import wavfile

SAMPLE_RATE = 16000
FIXED_LENGTH = 32000
model = tf.keras.models.load_model("watermark_denoiser.h5")

def normalize_audio(audio):
    print(audio / np.max(np.abs(audio)))
    return audio / np.max(np.abs(audio))

def pad_or_trim(audio):
    if len(audio) < FIXED_LENGTH:
        return np.pad(audio, (0, FIXED_LENGTH - len(audio)))
    return audio[:FIXED_LENGTH]

def add_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)

def denoise(audio):
    """Denoise a single chunk (fixed-length input)."""
    assert len(audio) == FIXED_LENGTH, f"Input must be {FIXED_LENGTH} samples."
    x = audio.astype(np.float32)
    if np.max(np.abs(x)) > 0:
        x /= np.max(np.abs(x))  # Normalize to [-1, 1]
    x = x[np.newaxis, ..., np.newaxis]  # Reshape for model
    denoised = model.predict(x, verbose=0)[0].squeeze()
    return denoised

def denoise_long_audio(audio, chunk_size=FIXED_LENGTH):
    """Denoise long audio by splitting into fixed-length chunks."""
    # Split into chunks (2-second segments)
    chunks = [
        audio[i:i + chunk_size] 
        for i in range(0, len(audio), chunk_size)
    ]
    
    # Denoise each chunk
    denoised_chunks = []
    for chunk in chunks:
        # Pad the last chunk if shorter than chunk_size
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        denoised_chunk = denoise(chunk)  # Use existing denoise() function
        denoised_chunks.append(denoised_chunk)
    
    # Combine chunks and trim to original length
    denoised_audio = np.concatenate(denoised_chunks)[:len(audio)]
    return denoised_audio

def process_audio(input_file, output_dir):
    rate, data = wav.read(input_file)
    print(len(data))
    print(data.ndim)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = normalize_audio(data)
    noisy = add_noise(data)
    denoised = denoise_long_audio(noisy)

    os.makedirs(output_dir, exist_ok=True)
    wav.write(os.path.join(output_dir, "original.wav"), rate, np.int16(data * 32767))
    wav.write(os.path.join(output_dir, "noisy.wav"), rate, np.int16(noisy * 32767))
    wav.write(os.path.join(output_dir, "denoised.wav"), rate, np.int16(denoised * 32767))

    print("Saved in:", output_dir)

    # Load original audio (30 seconds = 480,000 samples at 16kHz)
    #sample_rate, original_audio = wavfile.read("your_audio.wav")
    #original_audio = original_audio.astype(np.float32)

# Denoise (output will match original length)
    #denoised_audio = denoise_long_audio(original_audio)

# Verify lengths
    #print(f"Original: {len(original_audio)} samples")  # 480,000
    #print(f"Denoised: {len(denoised_audio)} samples")  # 480,000

# Example usage
process_audio("output audio low.wav", "processed_audio")
