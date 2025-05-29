# find_mask.py

import numpy as np

def find_mask(X, fs):
    """
    Calculate the masking threshold for audio watermarking
    
    Args:
        X: FFT coefficients
        fs: Sampling frequency
    
    Returns:
        MR: Masking threshold
    """
    N = len(X)
    data = 20 * np.log10(np.abs(X) + 1e-10)
    
    # Find peaks
    peaks = []
    for i in range(1, N-1):
        if (data[i] > data[i-1] and data[i] >= data[i+1] and data[i] > 0):
            peaks.append([data[i], i])
    peaks = np.array(peaks) if peaks else np.array([])
    
    df = fs/N
    N = N//2
    MR = np.zeros(N)
    
    # Calculate masking threshold
    for i in range(N):
        if i == 0:
            MR[i] = 0
            continue
        zi = 13 * np.arctan(0.00076 * i * df) + 3.5 * np.arctan(((i * df)/7500)**2)
        MR[i] = (3.64 * (i*df/1000)**(-0.8) - 
                 6.5 * np.exp(-0.6 * ((i*df/1000)-3.3)**2) + 
                 (1e-3) * (i*df/1000)**4 - 10)
        
        if len(peaks) > 0:
            for j in range(len(peaks)):
                k = int(peaks[j, 1])
                amp = peaks[j, 0]
                zk = 13 * np.arctan(0.00076 * k * df) + 3.5 * np.arctan(((k * df)/7500)**2)
                dz = zi - zk
                M_tmp = MR[i]
                
                if -3 <= dz < 8:
                    avtm = -1.525 - 0.275 * zk - 4.5
                    if -3 <= dz < -1:
                        vf = 17 * (dz + 1) - (0.4 * amp + 6)
                    elif -1 <= dz < 0:
                        vf = (0.4 * amp + 6) * dz
                    elif 0 <= dz < 1:
                        vf = -17 * dz
                    elif 1 <= dz < 8:
                        vf = -(dz - 1) * (17 - 0.15 * amp) - 17
                    
                    M_tmp = amp + avtm + vf
                
                if M_tmp > MR[i]:
                    MR[i] = M_tmp
                if MR[i] > 100:
                    MR[i] = 70
    
    return MR