# noise_suppression.py
import numpy as np
from scipy.signal import get_window

def reduce_noise(audio_data, sample_rate, noise_level=1.0, method='multiband'):
    """
    Khử tiếng ồn với 2 thuật toán:

      'multiband' (mặc định — tốt hơn):
          Chia phổ thành 4 band tần số, xử lý độc lập từng band.
          Adaptive: cập nhật profile nhiễu liên tục từ các frame yên tĩnh.
          → Hiệu quả với nhiễu thay đổi theo thời gian (gió, phòng ồn).

      'wiener':
          Wiener Filter cổ điển — mượt nhưng kém với nhiễu không ổn định.
    """
    print(f"Đang khử ồn [{method.upper()}] mức độ {noise_level:.0%}...")

    if np.max(np.abs(audio_data)) == 0:
        return audio_data

    n_fft      = 1024
    hop_length = n_fft // 4
    window     = get_window('hann', n_fft)

    frames          = _frame_signal(audio_data, n_fft, hop_length)
    windowed_frames = frames * window[:, np.newaxis]
    stft_matrix     = np.fft.rfft(windowed_frames, axis=0)

    magnitude = np.abs(stft_matrix)
    phase     = np.angle(stft_matrix)

    if method == 'multiband':
        magnitude_clean = _multiband_subtraction(
            magnitude, sample_rate, n_fft, noise_level)
    else:
        magnitude_clean = _wiener_filter(
            magnitude, sample_rate, n_fft, noise_level)

    magnitude_clean = _smooth_frames(magnitude_clean)

    stft_clean = magnitude_clean * np.exp(1j * phase)
    output     = _istft(stft_clean, n_fft, hop_length, window, len(audio_data))

    print("Khử ồn hoàn tất!")
    return output.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
#  MULTIBAND ADAPTIVE SPECTRAL SUBTRACTION
# ══════════════════════════════════════════════════════════════════

def _multiband_subtraction(magnitude, sample_rate, n_fft, noise_level):
    """
    Chia phổ thành 4 band, xử lý độc lập:
      Band 0:   0 –  300 Hz  (rumble, hum)       → trừ mạnh hơn
      Band 1: 300 – 2000 Hz  (giọng nói cốt lõi) → trừ nhẹ, bảo toàn giọng
      Band 2: 2k  – 6k  Hz  (sibilance, detail)  → trừ vừa
      Band 3: 6k  – Nyq  Hz  (hiss, air noise)    → trừ mạnh

    Adaptive noise profile:
      Cập nhật ước lượng nhiễu theo thời gian từ các frame yên tĩnh
      (frame có energy thấp hơn ngưỡng) → theo kịp nhiễu thay đổi.
    """
    freq_bins = magnitude.shape[0]
    n_frames  = magnitude.shape[1]
    nyquist   = sample_rate / 2

    # Tính bin tương ứng với từng band
    def hz_to_bin(hz):
        return int(hz / nyquist * (freq_bins - 1))

    bands = [
        (0,              hz_to_bin(300),  noise_level * 1.5, 0.01),  # rumble — trừ mạnh
        (hz_to_bin(300), hz_to_bin(2000), noise_level * 0.7, 0.05),  # giọng — trừ nhẹ
        (hz_to_bin(2000),hz_to_bin(6000), noise_level * 1.0, 0.02),  # detail — vừa
        (hz_to_bin(6000),freq_bins,        noise_level * 1.3, 0.01),  # hiss — trừ mạnh
    ]

    # ── Adaptive noise profile ────────────────────────────────────
    # Khởi tạo từ 10% đầu file
    init_frames     = max(1, n_frames // 10)
    noise_profile   = np.mean(magnitude[:, :init_frames], axis=1, keepdims=True)
    adaptive_profile = noise_profile.copy()

    # Ngưỡng energy để xác định frame yên tĩnh
    frame_energy     = np.mean(magnitude ** 2, axis=0)
    energy_threshold = np.percentile(frame_energy, 30)  # 30% frame thấp nhất = yên tĩnh

    magnitude_clean = magnitude.copy()

    for i in range(n_frames):
        # Cập nhật profile nếu frame này yên tĩnh
        if frame_energy[i] < energy_threshold:
            alpha = 0.95  # smoothing factor — cập nhật chậm, ổn định
            adaptive_profile[:, 0] = (
                alpha * adaptive_profile[:, 0] +
                (1 - alpha) * magnitude[:, i]
            )

        # Xử lý từng band độc lập
        for (b_start, b_end, alpha_sub, floor_ratio) in bands:
            if b_start >= b_end:
                continue
            sig  = magnitude[b_start:b_end, i]
            prof = adaptive_profile[b_start:b_end, 0]
            floor = floor_ratio * prof

            cleaned = sig - alpha_sub * prof
            cleaned = np.maximum(cleaned, floor)
            magnitude_clean[b_start:b_end, i] = cleaned

    return magnitude_clean


# ══════════════════════════════════════════════════════════════════
#  WIENER FILTER (giữ lại làm option)
# ══════════════════════════════════════════════════════════════════

def _wiener_filter(magnitude, sample_rate, n_fft, noise_level):
    """Wiener Filter cổ điển — noise profile tĩnh từ 10% đầu file."""
    n_frames      = magnitude.shape[1]
    init_frames   = max(1, n_frames // 10)
    noise_profile = np.mean(magnitude[:, :init_frames], axis=1, keepdims=True)

    noise_power  = (noise_profile * noise_level) ** 2
    signal_power = magnitude ** 2
    snr_est      = np.maximum(signal_power - noise_power, 0) / (noise_power + 1e-10)
    gain         = snr_est / (snr_est + 1.0)
    gain         = np.maximum(gain, 0.02)

    return magnitude * gain


# ══════════════════════════════════════════════════════════════════
#  HÀM PHỤ TRỢ
# ══════════════════════════════════════════════════════════════════

def _smooth_frames(magnitude, window_size=3):
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