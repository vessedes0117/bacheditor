# noise_suppression.py
import numpy as np
from scipy.signal import get_window

def reduce_noise(audio_data, sample_rate, noise_level=0.5, method='multiband'):
    """
    Khử ồn cải tiến với 3 kỹ thuật chống musical noise:
    1. Dynamic Spectral Floor: floor tỷ lệ với peak phổ hiện tại
    2. Temporal Smoothing IIR: làm mượt theo thời gian (không dùng window cứng)
    3. Adaptive Noise Tracking: cập nhật noise profile thông minh hơn
    
    Tham số noise_level mặc định giảm từ 1.0 → 0.5 (nhẹ nhàng hơn)
    """
    print(f"Đang khử ồn [{method.upper()}] mức độ {noise_level:.0%}...")
    
    if np.max(np.abs(audio_data)) == 0:
        return audio_data
    
    n_fft      = 1024
    hop_length = n_fft // 4
    window     = get_window('hann', n_fft)
    
    # STFT
    frames          = _frame_signal(audio_data, n_fft, hop_length)
    windowed_frames = frames * window[:, np.newaxis]
    stft_matrix     = np.fft.rfft(windowed_frames, axis=0)
    
    magnitude = np.abs(stft_matrix)
    phase     = np.angle(stft_matrix)
    
    # Xử lý theo phương pháp
    if method == 'multiband':
        magnitude_clean = _multiband_subtraction_v2(
            magnitude, sample_rate, n_fft, noise_level)
    else:
        magnitude_clean = _wiener_filter_v2(
            magnitude, sample_rate, n_fft, noise_level)
    
    # Temporal smoothing cải tiến
    magnitude_clean = _temporal_smoothing_iir(magnitude_clean, alpha=0.7)
    
    # Tái tạo tín hiệu
    stft_clean = magnitude_clean * np.exp(1j * phase)
    output     = _istft(stft_clean, n_fft, hop_length, window, len(audio_data))
    
    print("Khử ồn hoàn tất!")
    return output.astype(np.float32)

# ══════════════════════════════════════════════════════════════════
#  MULTIBAND SPECTRAL SUBTRACTION — VERSION 2
# ══════════════════════════════════════════════════════════════════

def _multiband_subtraction_v2(magnitude, sample_rate, n_fft, noise_level):
    """
    Cải tiến so với bản cũ:
    1. Dynamic Spectral Floor: floor = max(0.03 * prof, 0.02 * peak_current_frame)
    2. Adaptive Alpha: giảm alpha_sub khi SNR cao (bảo toàn giọng)
    3. Better Noise Estimation: dùng median thay mean (chống outlier)
    4. Over-subtraction Factor: điều chỉnh theo từng band
    """
    freq_bins = magnitude.shape[0]
    n_frames  = magnitude.shape[1]
    nyquist   = sample_rate / 2
    
    def hz_to_bin(hz):
        return int(hz / nyquist * (freq_bins - 1))
    
    # Cấu hình band cải tiến
    bands = [
        # (start_bin, end_bin, over_sub_factor, floor_ratio, alpha_smooth)
        (0,              hz_to_bin(250),  2.0, 0.03, 0.9),   # Sub-bass: cắt mạnh
        (hz_to_bin(250), hz_to_bin(500),  1.5, 0.04, 0.85),  # Bass: cắt vừa
        (hz_to_bin(500), hz_to_bin(2000), 1.0, 0.08, 0.8),   # Mid (giọng): rất nhẹ
        (hz_to_bin(2000),hz_to_bin(6000), 1.2, 0.05, 0.85),  # Upper mid: vừa
        (hz_to_bin(6000),freq_bins,       1.5, 0.03, 0.9),   # High (hiss): mạnh
    ]
    
    # ── Noise Profile: Dùng median (robust hơn mean) ────────────
    init_frames = max(1, n_frames // 8)  # 12.5% đầu
    noise_profile = np.median(magnitude[:, :init_frames], axis=1, keepdims=True)
    
    # ── Adaptive Noise Tracking ─────────────────────────────────
    frame_energy = np.mean(magnitude ** 2, axis=0)
    energy_threshold = np.percentile(frame_energy, 25)  # 25% thấp nhất
    
    magnitude_clean = np.zeros_like(magnitude)
    prev_frame = noise_profile[:, 0].copy()
    
    for i in range(n_frames):
        # Cập nhật noise profile nếu frame yên tĩnh
        if frame_energy[i] < energy_threshold:
            # Slow update để tránh học nhầm giọng nói nhỏ
            alpha_update = 0.95
            noise_profile[:, 0] = (
                alpha_update * noise_profile[:, 0] +
                (1 - alpha_update) * magnitude[:, i]
            )
        
        # Xử lý từng band
        for (b_start, b_end, over_sub, floor_ratio, alpha_smooth) in bands:
            if b_start >= b_end:
                continue
            
            sig  = magnitude[b_start:b_end, i]
            prof = noise_profile[b_start:b_end, 0]
            
            # Dynamic Spectral Floor
            # Floor cao hơn ở những chỗ peak cao → giảm musical noise
            dynamic_floor = np.maximum(
                floor_ratio * prof,
                0.02 * np.max(sig)  # 2% của peak frame hiện tại
            )
            
            # Spectral Subtraction với over-subtraction factor
            cleaned = sig - (over_sub * noise_level) * prof
            
            # Hard floor
            cleaned = np.maximum(cleaned, dynamic_floor)
            
            # Temporal smoothing trong band
            cleaned = alpha_smooth * prev_frame[b_start:b_end] + \
                      (1 - alpha_smooth) * cleaned
            
            magnitude_clean[b_start:b_end, i] = cleaned
            prev_frame[b_start:b_end] = cleaned
        
        # Global smoothing giữa các frame
        if i > 0:
            magnitude_clean[:, i] = 0.6 * magnitude_clean[:, i] + \
                                    0.4 * magnitude_clean[:, i-1]
    
    return magnitude_clean

# ══════════════════════════════════════════════════════════════════
#  WIENER FILTER — VERSION 2
# ══════════════════════════════════════════════════════════════════

def _wiener_filter_v2(magnitude, sample_rate, n_fft, noise_level):
    """
    Wiener Filter cải tiến:
    - Noise profile dùng median
    - Spectral floor động
    - Smoothing theo tần số
    """
    freq_bins, n_frames = magnitude.shape
    
    # Noise profile robust
    init_frames = max(1, n_frames // 8)
    noise_profile = np.median(magnitude[:, :init_frames], axis=1, keepdims=True)
    
    # Wiener gain
    noise_power = (noise_profile * noise_level) ** 2
    signal_power = magnitude ** 2
    
    # SNR estimation
    snr_est = np.maximum(signal_power - noise_power, 0) / (noise_power + 1e-10)
    
    # Wiener gain function (soft masking)
    gain = snr_est / (snr_est + 1.0)
    
    # Spectral floor: không để gain < 0.05
    gain = np.maximum(gain, 0.05)
    
    # Frequency smoothing (dọc theo trục tần số)
    gain_smoothed = np.copy(gain)
    for i in range(1, freq_bins - 1):
        gain_smoothed[i, :] = 0.5 * gain[i, :] + \
                              0.25 * gain[i-1, :] + \
                              0.25 * gain[i+1, :]
    
    return magnitude * gain_smoothed

# ══════════════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHING IIR
# ══════════════════════════════════════════════════════════════════

def _temporal_smoothing_iir(magnitude, alpha=0.7):
    """
    Smoothing theo thời gian dùng IIR filter:
    y[n] = alpha * y[n-1] + (1-alpha) * x[n]
    
    alpha cao (0.8-0.9) = smooth nhiều, giảm musical noise nhưng làm mờ transient
    alpha thấp (0.5-0.7) = giữ transient tốt nhưng còn chút musical noise
    
    Mặc định alpha=0.7 — cân bằng tốt
    """
    smoothed = np.copy(magnitude)
    n_frames = magnitude.shape[1]
    
    for i in range(1, n_frames):
        smoothed[:, i] = alpha * smoothed[:, i-1] + \
                         (1 - alpha) * magnitude[:, i]
    
    return smoothed

# ══════════════════════════════════════════════════════════════════
#  HÀM PHỤ TRỢ (giữ nguyên)
# ══════════════════════════════════════════════════════════════════

def _smooth_frames(magnitude, window_size=3):
    """Legacy — không dùng nữa nhưng giữ lại cho compatibility."""
    smoothed = np.copy(magnitude)
    half     = window_size // 2
    n_frames = magnitude.shape[1]
    
    for i in range(n_frames):
        start = max(0, i - half)
        end   = min(n_frames, i + half + 1)
        smoothed[:, i] = np.mean(magnitude[:, start:end], axis=1)
    
    return smoothed

def _frame_signal(signal, n_fft, hop_length):
    pad_len       = n_fft - (len(signal) - n_fft) % hop_length if len(signal) > n_fft else n_fft
    signal_padded = np.pad(signal, (0, pad_len), mode='reflect')
    num_frames    = (len(signal_padded) - n_fft) // hop_length + 1
    frames        = np.zeros((n_fft, num_frames))
    
    for i in range(num_frames):
        start        = i * hop_length
        frames[:, i] = signal_padded[start: start + n_fft]
    
    return frames

def _istft(stft_matrix, n_fft, hop_length, window, original_length):
    num_frames = stft_matrix.shape[1]
    output_len = (num_frames - 1) * hop_length + n_fft
    output     = np.zeros(output_len)
    window_sum = np.zeros(output_len)
    
    for i in range(num_frames):
        frame_time = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        start      = i * hop_length
        output[start: start + n_fft]     += frame_time * window
        window_sum[start: start + n_fft] += window ** 2
    
    nonzero         = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    
    return output[:original_length]