import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== UNet with Skip Connections ===================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super().__init__()
        self.down1 = self.conv_block(in_channels, base_channels)
        self.down2 = self.conv_block(base_channels, base_channels * 2)
        self.down3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.down4 = self.conv_block(base_channels * 4, base_channels * 8)

        self.up3 = self.upconv_block(base_channels * 8, base_channels * 4)
        self.up2 = self.upconv_block(base_channels * 4, base_channels * 2)
        self.up1 = self.upconv_block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, x):
        original_shape = x.shape
        x1 = self.down1(x)  # 64
        x2 = self.down2(x1) # 128
        x3 = self.down3(x2) # 256
        x4 = self.down4(x3) # 512

        x = self.up3(x4)
        x = match_tensor(x, x3)
        x = x + x3
        x = self.up2(x)
        x = match_tensor(x, x2)
        x = x + x2
        x = self.up1(x)
        x = match_tensor(x, x1)
        if x.shape != x1.shape:
            diff = x1.shape[-1] - x.shape[-1]
            if diff > 0:
                x1 = x1[..., :-diff]
            elif diff < 0:
                x1 = F.pad(x1, (0, -diff))  # pad if x1 is shorter
        x = x + x1

        x = self.final(x)
        if x.shape != original_shape:
            x = F.interpolate(x, size=original_shape[-2:], mode='bilinear', align_corners=False)
        return x

def match_tensor(x, ref):
    # Pads or crops x to match ref shape
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)

        if diffY > 0 or diffX > 0:
            x = F.pad(x, [0, diffX, 0, diffY])
        elif diffY < 0 or diffX < 0:
            x = x[:, :, :ref.size(2), :ref.size(3)]

        return x
# =================== Audio Dataset ===================
class AudioDataset(Dataset):
    def __init__(self, file_list, noise_level=0):
        self.file_list = file_list
        self.noise_level = noise_level
        self.transform = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=64)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, _ = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-6)

        mel = self.transform(waveform)
        mel = torch.log(mel + 1e-6)

        # Add controlled Gaussian noise
        noisy = mel + self.noise_level * torch.randn_like(mel)

        return noisy, mel

from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    noisy_list, clean_list = zip(*batch)

    # Squeeze channel dimension if needed or move time to first dim
    # Assume shape: [1, freq_bins, time] -> we want to pad along dim=2

    # Ensure same shape for all tensors before padding (pad time dimension)
    max_len = max(x.shape[-1] for x in noisy_list)

    def pad_tensor(t, max_len):
        pad_len = max_len - t.shape[-1]
        if pad_len > 0:
            return F.pad(t, (0, pad_len))  # pad time dimension
        return t  # already max_len

    noisy_padded = torch.stack([pad_tensor(x, max_len) for x in noisy_list])
    clean_padded = torch.stack([pad_tensor(x, max_len) for x in clean_list])

    return noisy_padded, clean_padded



# =================== Training Loop ===================
def train_model(dataset, batch_size=4, epochs=50, lr=1e-4):
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = DataLoader(audio_dataset, batch_size= 8, collate_fn=collate_fn)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "unet_audio_denoiser.pth")
    return model

# =================== Denoising ===================
def denoise_audio(model, file_path, noise_level=0):
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min() + 1e-6)

    mel_transform = torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=64)
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    noisy = mel + noise_level * torch.randn_like(mel)

    model.eval()
    with torch.no_grad():
        denoised = model(noisy.unsqueeze(0).to(device)).cpu().squeeze(0)

    # Invert mel-spectrogram
    mel_to_spec = torchaudio.transforms.InverseMelScale(n_stft=342, n_mels=64)
    spec = torch.exp(denoised)
    spec = torch.clamp(mel_to_spec(spec), min=1e-6)

    # Griffin-Lim to reconstruct waveform
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)
    waveform_recon = griffin_lim(spec)
    waveform_recon /= waveform_recon.abs().max()

    output_path = f"denoised_{os.path.basename(file_path)}"
    if waveform_recon.ndim == 1:
        waveform_recon = waveform_recon.unsqueeze(0)  # [1, time]
    elif waveform_recon.ndim == 3:
        waveform_recon = waveform_recon.squeeze(0)    # Remove batch dim
    # Now save
    torchaudio.save(output_path, waveform_recon, sr)

    #torchaudio.save(output_path, waveform_recon.unsqueeze(0), sr)
    print(f"Denoised audio saved to {output_path}")
    return waveform_recon

# =================== Main Execution ===================
if __name__ == "__main__":
    dataset_folder = r"C:\watermark-audio-diffusion-main\samp"
    file_list = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".wav")]

    if not file_list:
        print("No .wav files found.")
    else:
        print(f"Found {len(file_list)} files. Starting training...")
        audio_dataset = AudioDataset(file_list, noise_level=0.05)
        model = train_model(audio_dataset)

        for file in file_list:
            print(f"Denoising {file}...")
            denoise_audio(model, file)
