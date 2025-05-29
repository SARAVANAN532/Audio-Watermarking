import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import torch.nn.functional as F
import numpy as np

# UNet-based Diffusion Model for Denoising
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64):
        super(UNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, hidden_dim, 3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Conv2d(hidden_dim, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x1 = torch.relu(self.enc1(x))
        x2 = torch.relu(self.enc2(x1))
        x3 = torch.relu(self.enc3(x2))

        # Decoder
        x4 = torch.relu(self.dec1(x3))
        x5 = torch.relu(self.dec2(x4))
        x6 = self.dec3(x5)

        # Resize the output to match the input spectrogram size
        x6 = torch.nn.functional.interpolate(x6, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x6

# Function to add controlled noise
def add_noise(audio, noise_level=0):
    noise = noise_level * torch.randn_like(audio)
    return audio + noise, noise

# Function to preprocess and extract features
# Function to preprocess and extract features
def preprocess_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)

        # Ensure waveform is not empty
        if waveform.shape[0] == 0:
            print(f"Error: Audio file {file_path} is empty.")
            return None, None, None

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize the waveform (optional)
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())

        # Use MelSpectrogram with adjusted parameters
        n_mels = 64  # Further reduce n_mels
        n_fft = 1024  # Increase n_fft for better frequency resolution
        spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft)(waveform)

        # Check for NaNs or Infinities in the spectrogram
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print(f"Warning: Spectrogram contains NaN or Inf values in file {file_path}. Replacing with zeros.")
            spectrogram = torch.nan_to_num(spectrogram)  # Replace NaN with zero

        # Apply log scaling to the spectrogram to make values more manageable
        spectrogram = torch.log(spectrogram + 1e-6)  # Log scaling for better numerical stability

        # Ensure the width of the spectrogram is divisible by 2
        width_padding = (2 - spectrogram.shape[2] % 2) % 2  # Padding to make width divisible by 2
        spectrogram = F.pad(spectrogram, (0, width_padding))

        return spectrogram.unsqueeze(0), waveform, sample_rate  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing audio file {file_path}: {e}")
        return None, None, None

# Calculate PSNR
def calculate_psnr(original, denoised):
    mse = torch.mean((original - denoised) ** 2)
    max_value = torch.max(original)
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()

# Calculate NC (Normalized Cross-Correlation)
def calculate_nc(original, denoised):
    original = original.flatten()
    denoised = denoised.flatten()
    
    numerator = torch.sum(original * denoised)
    denominator = torch.sqrt(torch.sum(original ** 2) * torch.sum(denoised ** 2))
    
    nc = numerator / denominator
    return nc.item()

# Training loop
def train_diffusion_model(dataset, epochs=100, noise_level=0):
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    psnr_per_file = {}
    nc_per_file = {}
    loss_per_file = {}

    for epoch in range(epochs):
        epoch_psnr = 0
        epoch_nc = 0
        epoch_loss = 0
        total_files = 0

        for file in dataset:
            spectrogram, waveform, sample_rate = preprocess_audio(file)

            # Add noise
            noisy_spectrogram, _ = add_noise(spectrogram, noise_level)

            # Move to tensors
            spectrogram = spectrogram.to(torch.float32)
            noisy_spectrogram = noisy_spectrogram.to(torch.float32)

            # Train model
            optimizer.zero_grad()
            output = model(noisy_spectrogram)
            loss = criterion(output, spectrogram)
            loss.backward()
            optimizer.step()

            # Calculate PSNR and NC
            psnr = calculate_psnr(spectrogram, output)
            nc = calculate_nc(spectrogram, output)

            # Save PSNR, NC, and Loss per file
            if file not in psnr_per_file:
                psnr_per_file[file] = []
                nc_per_file[file] = []
                loss_per_file[file] = []

            psnr_per_file[file].append(psnr)
            nc_per_file[file].append(nc)
            loss_per_file[file].append(loss.item())

            epoch_psnr += psnr
            epoch_nc += nc
            epoch_loss += loss.item()
            total_files += 1

        avg_psnr = epoch_psnr / total_files
        avg_nc = epoch_nc / total_files
        avg_loss = epoch_loss / total_files

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, NC: {avg_nc:.4f}")

    # At the end of training, print PSNR, NC, and Loss per file
    print("\nFinal PSNR, NC, and Loss for each audio file:")
    for file in psnr_per_file:
        avg_psnr_file = np.mean(psnr_per_file[file])
        avg_nc_file = np.mean(nc_per_file[file])
        avg_loss_file = np.mean(loss_per_file[file])
        print(f"{file} - PSNR: {avg_psnr_file:.4f}, NC: {avg_nc_file:.4f}, Loss: {avg_loss_file:.4f}")



        

    torch.save(model.state_dict(), "trained_diffusion_model.pth")
    return model

# Function to denoise and reconstruct audio

def denoise_audio(model, file_path):
    spectrogram, waveform, sample_rate = preprocess_audio(file_path)
    if spectrogram is None:
        return None, None  # Early exit if preprocessing failed

    noisy_spectrogram, _ = add_noise(spectrogram)

    model.eval()
    with torch.no_grad():
        denoised_spectrogram = model(noisy_spectrogram.to(torch.float32))

    # Ensure correct dimensionality
    denoised_spectrogram = denoised_spectrogram.squeeze(0)  # Remove batch dimension
    denoised_spectrogram = denoised_spectrogram.squeeze()   # Ensure it's 2D

    # Check spectrogram shape
    print(f"Denoised Spectrogram Shape: {denoised_spectrogram.shape}")

    if len(denoised_spectrogram.shape) != 2:
        print("Error: The tensor passed to inverse transformation is not 2D!")
        return None, None

    # Ensure n_stft matches the value used in MelSpectrogram
    n_fft = 1024
    n_mels = 64
    n_stft = n_fft // 2 + 1

    inv_mel_transform = torchaudio.transforms.InverseMelScale(n_stft=n_stft, n_mels=n_mels)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft)

# 1. MelSpectrogram -> Spectrogram
    spectrogram = inv_mel_transform(denoised_spectrogram)

# 2. Spectrogram -> Waveform
    waveform_reconstructed = griffin_lim(spectrogram)

# Normalize
    waveform_reconstructed /= waveform_reconstructed.abs().max()
    waveform_int16 = (waveform_reconstructed * 32767).to(torch.int16)
        # Extract filename
    file_name = os.path.basename(file_path).replace('.wav', '')

        # Save denoised audio
    output_path = f"denoised_{file_name}.wav"
    if waveform_int16.ndim == 1:
        waveform_int16 = waveform_int16.unsqueeze(0)  # Add a channel dimension

    torchaudio.save(output_path, waveform_int16, sample_rate)
    print(f"Saved denoised audio as {output_path}")

    #except Exception as e:
       # print(f"Error during inverse transformation: {e}")
        #return None, None

    return waveform_reconstructed, sample_rate



# Dataset path
dataset_folder = r"C:\watermark-audio-diffusion-main\samp"  # Raw string prevents escape sequence issues

# Get all .wav files from the dataset folder
dataset = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith(".wav")]

if len(dataset) == 0:
    print("No .wav files found in the dataset folder.")
else:
    print(f"Found {len(dataset)} audio files.")

    # Train the diffusion model using all audio files in the dataset
    trained_model = train_diffusion_model(dataset)

    # Denoise all files one by one
    for file in dataset:
        print(f"Processing {file}...")
        denoised_audio, sr = denoise_audio(trained_model, file)

 