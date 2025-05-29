import numpy as np
import os
import tensorflow as tf
Input = tf.keras.layers.Input
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
UpSampling1D = tf.keras.layers.UpSampling1D
Model = tf.keras.Model
Adam = tf.keras.optimizers.Adam
from scipy.io import wavfile

SAMPLE_RATE = 16000
FIXED_LENGTH = 32000  # 2 seconds

def load_audio(file_path):
    rate, data = wavfile.read(file_path)
    if data.ndim > 1:  # Convert stereo to mono
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    
    # Normalize to [-1, 1]
    if np.max(np.abs(data)) > 0:
        data /= np.max(np.abs(data))
    
    # Pad or truncate to FIXED_LENGTH (for training only)
    if len(data) < FIXED_LENGTH:
        data = np.pad(data, (0, FIXED_LENGTH - len(data)))
    else:
        data = data[:FIXED_LENGTH]  # Truncate if too long
    
    return data

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return np.clip(data + noise, -1.0, 1.0)

# Load training data
clean_dir = "clean_watermarked_audios"
X_train = []
Y_train = []

for file in os.listdir(clean_dir):
    if file.endswith(".wav"):
        audio = load_audio(os.path.join(clean_dir, file))
        assert len(audio) == FIXED_LENGTH, f"File {file} has length {len(audio)} (expected {FIXED_LENGTH})"
        clean = load_audio(os.path.join(clean_dir, file))
        noisy = add_noise(clean)
        X_train.append(noisy)
        Y_train.append(clean)

X_train = np.array(X_train)[..., np.newaxis]
Y_train = np.array(Y_train)[..., np.newaxis]

# Build 1D convolutional autoencoder
input_audio = Input(shape=(FIXED_LENGTH, 1))
x = Conv1D(16, 9, activation='relu', padding='same')(input_audio)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(8, 9, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(8, 9, activation='relu', padding='same')(x)

x = UpSampling1D(2)(x)
x = Conv1D(8, 9, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 9, activation='tanh', padding='same')(x)

model = Model(input_audio, decoded)
model.compile(optimizer=Adam(1e-4), loss='mean_squared_error')
model.summary()

# Train
model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_split=0)

# Save model
model.save("watermark_denoiser.h5")
