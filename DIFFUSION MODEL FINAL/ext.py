# extract_watermark.py
import os
import numpy as np
from scipy.io import wavfile
from PIL import Image
from scipy.io import wavfile
import torchaudio
from scipy.stats import pearsonr
from find_mask import find_mask

def extract_watermark(audio_path, h, w, output_path):
    """
    Extract watermark from audio file
    
    Args:
        audio_path: Path to watermarked audio file
        h: Height of original image
        w: Width of original image
        output_path: Path to save extracted watermark
    """
    # Read audio file
    x, fs = torchaudio.load(audio_path)
    print(x)
    x = x - x.mean()  # Center it around 0
    x = x / x.abs().max()  # Normalize to -1 to 1

    #desired_sr = 16000
    
    #if x.shape[0] > 1:
        #x = x.mean(dim=0, keepdim=True)
    
    #if fs != desired_sr:
        #resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=desired_sr)
        #x = resampler(x)
        #fs = desired_sr

    #x = x.squeeze().numpy()  # Convert to 1D NumPy array for FFT

    L = len(x)
    print(L)
    N = 512
    num_frame = L // N
    
    # Initialize extracted image
    im = np.zeros((h, w, 3), dtype=np.uint8)
    row = 0
    column = 0
    mask_no_ex = np.zeros(num_frame)
    print(num_frame)
    print(os.path.exists(audio_path))  # Should be True
    print(audio_path)
    for mm in range(num_frame):
        tt = slice(mm * N, (mm + 1) * N)
        X = np.fft.fft(x[tt], N)
        M = find_mask(X, fs)
        print(f"Bits collected: {len(imsg)}, Frame {mm}")
        pix = np.zeros((1, 12, 3), dtype=np.uint8)
        row2 = 0
        column2 = 0
        color = 0
        count = 0
        imsg = ""
        count2 = 0
        
        for k in range(N // 2):
            check = 0
            print(1)
            print(f"FFT Mag: {np.abs(X[k])}, Mask: {M[k]}, dB: {20 * np.log10(np.abs(X[k]) + 1e-10)}")

            if 20 * np.log10(np.abs(X[k]) + 1e-10) < M[k]:
                count2 += 1
                if np.abs(X[k]) < np.exp(-1.8):
                    check = 1
                
                if np.abs(X[k]) >= np.exp(-2.1):
                    imsg_tmp1, imsg_tmp2 = '1', '1'
                elif np.abs(X[k]) >= np.exp(-4.1):
                    imsg_tmp1, imsg_tmp2 = '1', '0'
                elif np.abs(X[k]) >= np.exp(-6.1):
                    imsg_tmp1, imsg_tmp2 = '0', '1'
                else:
                    imsg_tmp1, imsg_tmp2 = '0', '0'
                
                if check == 1:
                    imsg += imsg_tmp1 + imsg_tmp2
                    if len(imsg) >= 8:
                        byte = imsg[:8]
                        imsg = imsg[8:]
                        if row2 < pix.shape[0] and column2 < pix.shape[1]:
                            pix[row2, column2, color] = int(byte, 2)
                        color += 1
                        if color >= 3:
                            color = 0
                            column2 += 1
                            if column2 >= pix.shape[1]:
                                column2 = 0
                                row2 += 1
        
        # Update image with extracted pixels
        if (column + 12) > w:
            rem = w - column
            im[row, column:w, :] = pix[0, :rem, :]
            row += 1
            column = 0
            if row < h:
                im[row, column:column + (12 - rem), :] = pix[0, rem:12, :]
                column += (12 - rem)
        else:
            im[row, column:column + 12, :] = pix[0, :, :]
            column += 12
            if column >= w:
                column = 0
                row += 1

        mask_no_ex[mm] = count2

    # Save extracted image
    Image.fromarray(im).save(output_path)
    return im

def calculate_metrics(original_audio_path, watermarked_audio_path, 
                      original_image_path, extracted_image_path):
    """
    Calculate quality metrics for the watermarking system
    """
    desired_sr = 16000

    # Load and resample audio
    aud3, fs1 = wavfile.read(original_audio_path)
    #if aud3.shape[0] > 1:
        #aud3 = wavfile.read(dim=0, keepdim=True)
    #if fs1 != desired_sr:
        #resampler = torchaudio.transforms.Resample(orig_freq=fs1, new_freq=desired_sr)
        #aud3 = resampler(aud3)

    masked, fs2 = wavfile.read(watermarked_audio_path)
    #if masked.shape[0] > 1:
        #masked = masked.mean(dim=0, keepdim=True)
    #if fs2 != desired_sr:
        #resampler = torchaudio.transforms.Resample(orig_freq=fs2, new_freq=desired_sr)
        #masked = resampler(masked)

    # Ensure same length
    min_length = min(aud3.shape[1], masked.shape[1])
    aud3 = aud3[:, :min_length]
    masked = masked[:, :min_length]

    # Image PSNR
    original_img = np.array(Image.open(original_image_path))
    extracted_img = np.array(Image.open(extracted_image_path))

    if original_img.shape != extracted_img.shape:
        raise ValueError('Original and extracted images must be the same size')

    mse = np.mean((original_img.astype(float) - extracted_img.astype(float)) ** 2)
    psnr_value = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)

    print(f'PSNR between the original and extracted image: {psnr_value:.2f} dB')

    return psnr_value

# Example usage
if __name__ == "__main__":
    audio_path = "denoised_embed audio mid.wav"
    output_image_path = "output img.jpg"
    h, w = 170, 170

    # Extract watermark
    extracted_image = extract_watermark(audio_path, h, w, output_image_path)

    # Calculate metrics
    psnr = calculate_metrics(
        original_audio_path="embed audio mid.wav",
        watermarked_audio_path="denoised_embed audio mid.wav",
        original_image_path="input img1.jpg",
        extracted_image_path=output_image_path
    )
