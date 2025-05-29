import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UNet-based Diffusion Model for Denoising
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_dim, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 3, stride=2, padding=1)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.Conv2d(hidden_dim, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(x1))
        x3 = torch.relu(self.enc3(x2))
        x4 = torch.relu(self.enc4(x3))

        # Decoder
        x5 = torch.relu(self.dec1(x4))
        x6 = torch.relu(self.dec2(x5))
        x7 = torch.relu(self.dec3(x6))
        x8 = self.dec4(x7)

        # Final interpolation to match original size
        x8 = F.interpolate(x8, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x8

# Add controlled noise
def add_noise(audio, noise_level=0):
    #print(torch.randn_like(audio))
    #noise = noise_level * torch.randn_like(audio)
    noise = torch.normal(mean=0.0, std=noise_level, size=audio.shape)
    #print(audio + noise)
    return audio + noise, noise

# Preprocessing function
def preprocess_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] == 0:
            print(f"Error: Audio file {file_path} is empty.")
            return None, None, None

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-6)

        n_mels = 128
        n_fft = 1024
        mel_spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft)(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)

        width_padding = (2 - mel_spec.shape[2] % 2) % 2
        mel_spec = F.pad(mel_spec, (0, width_padding))

        return mel_spec.unsqueeze(0), waveform, sample_rate
    except Exception as e:
        print(f"Error preprocessing {file_path}: {e}")
        return None, None, None

# PSNR Calculation
def calculate_psnr(original, denoised):
    mse = torch.mean((original - denoised) ** 2)
    max_val = torch.max(original)
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-6))
    return psnr.item()

# Normalized Cross Correlation
def calculate_nc(original, denoised):
    original = original.flatten()
    denoised = denoised.flatten()
    return torch.dot(original, denoised) / (torch.norm(original) * torch.norm(denoised) + 1e-6)

# Training Loop
def train_diffusion_model(dataset, epochs=50, noise_level=0):
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = total_psnr = total_nc = 0
        for file in dataset:
            spectrogram, waveform, _ = preprocess_audio(file)
            #print(waveform)
            if spectrogram is None:
                continue

            noisy_spec, _ = add_noise(spectrogram, noise_level)
            spectrogram, noisy_spec = spectrogram.to(device), noisy_spec.to(device)

            optimizer.zero_grad()
            output = model(noisy_spec)
            loss = criterion(output, spectrogram)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr += calculate_psnr(spectrogram.cpu(), output.cpu())
            total_nc += calculate_nc(spectrogram.cpu(), output.cpu())

        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataset):.4f}, PSNR={total_psnr/len(dataset):.4f}, NC={total_nc/len(dataset):.4f}")

    torch.save(model.state_dict(), "trained_diffusion_model.pth")
    return model

# Denoising and reconstruction
def denoise_audio(model, file_path):
    spectrogram, waveform, sample_rate = preprocess_audio(file_path)
    if spectrogram is None:
        return None, None

    noisy_spec, _ = add_noise(spectrogram)
    model.eval()
    with torch.no_grad():
        denoised_spec = model(noisy_spec.to(device)).cpu().squeeze(0)

    n_fft = 1024
    n_mels = 128
    n_stft = n_fft // 2 + 1

    # Convert back to linear scale and spectrogram
    mel_spec_exp = torch.exp(denoised_spec.squeeze(0))
    mel_to_spec = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=n_mels)
    spec = mel_to_spec(mel_spec_exp)
    spec = torch.clamp(spec, min=1e-6)

    # Griffin-Lim to get waveform
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
    waveform_recon = griffin_lim(spec)

    # Normalize
    waveform_recon /= waveform_recon.abs().max()
    waveform_int16 = (waveform_recon * 32767).to(torch.int16)

    if waveform_int16.ndim == 1:
        waveform_int16 = waveform_int16.unsqueeze(0)

    output_path = f"denoised_{os.path.basename(file_path)}"
    torchaudio.save(output_path, waveform_int16, sample_rate)
    print(f"Saved denoised audio to {output_path}")
    return waveform_recon, sample_rate

# Main Execution
dataset_folder = r"C:\watermark-audio-diffusion-main\samp"
dataset = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".wav")]

if not dataset:
    print("No .wav files found.")
else:
    print(f"Found {len(dataset)} audio files.")
    model = train_diffusion_model(dataset)

    for file in dataset:
        print(f"Denoising {file}...")
        denoise_audio(model, file)
