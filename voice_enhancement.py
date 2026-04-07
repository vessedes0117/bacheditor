# voice_enhancement.py
import numpy as np
from scipy.signal import butter, filtfilt, iirpeak, iirnotch

# ══════════════════════════════════════════════════════════════════
#  HÀM CÔNG KHAI
# ══════════════════════════════════════════════════════════════════

def enhance_voice(audio_data, sample_rate,
                  bass_gain=0.0, mid_gain=1.0, treble_gain=0.0,
                  presence_boost=True, de_essing=True):
    """
    Tăng cường giọng nói qua 4 tầng xử lý nối tiếp:

      Tầng 1 — High-pass 80Hz:
        Cắt rumble/hum dưới 80Hz.

      Tầng 2 — EQ 3 band (Bass / Mid / Treble):
        Bass   :  80 – 300Hz  — âm trầm (điều chỉnh ±)
        Mid    : 300 – 3kHz   — dải giọng nói cốt lõi
        Treble : 3k  – 8kHz   — độ sáng, sắc nét

      Tầng 3 — Presence boost (2–5kHz):
        Tăng nhẹ dải 2–5kHz giúp giọng nổi bật, rõ ràng hơn
        trong môi trường nhiều tạp âm.

      Tầng 4 — De-essing (6–10kHz):
        Giảm âm "s", "ch" bị chói (sibilance).
        Dùng notch filter tại 7.5kHz — tần số sibilance phổ biến nhất.

    Tham số gain: 0.0 = tắt hoàn toàn, 1.0 = bình thường, >1.0 = boost
    """
    print("Đang tăng cường giọng nói...")

    audio = audio_data.copy().astype(np.float64)
    original_peak = np.max(np.abs(audio)) + 1e-10

    # ── Tầng 1: High-pass 80Hz ────────────────────────────────────
    audio = _apply_highpass(audio, cutoff=80.0, fs=sample_rate)

    # ── Tầng 2: EQ 3 band ────────────────────────────────────────
    audio = _apply_3band_eq(audio, sample_rate,
                            bass_gain, mid_gain, treble_gain)

    # ── Tầng 3: Presence boost ────────────────────────────────────
    if presence_boost:
        audio = _apply_presence_boost(audio, sample_rate)

    # ── Tầng 4: De-essing ─────────────────────────────────────────
    if de_essing:
        audio = _apply_deessing(audio, sample_rate)

    # ── Normalize về biên độ gốc ──────────────────────────────────
    current_peak = np.max(np.abs(audio)) + 1e-10
    audio        = audio * (original_peak / current_peak)

    print("Tăng cường giọng nói hoàn tất!")
    return audio.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
#  TẦNG 1 — HIGH-PASS
# ══════════════════════════════════════════════════════════════════

def _apply_highpass(audio, cutoff, fs, order=5):
    """Cắt tần số dưới cutoff Hz — loại bỏ rumble/hum."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, audio)


# ══════════════════════════════════════════════════════════════════
#  TẦNG 2 — EQ 3 BAND
# ══════════════════════════════════════════════════════════════════

def _apply_3band_eq(audio, fs, bass_gain, mid_gain, treble_gain):
    """
    EQ 3 band bằng cách tách phổ thành 3 dải rồi trộn lại có trọng số.

    Kỹ thuật: Linkwitz-Riley crossover đơn giản
      - Low  band  : lowpass  tại 300Hz
      - High band  : highpass tại 3000Hz
      - Mid  band  : phần còn lại (tổng - low - high)

    Trộn lại: out = low*bass + mid*mid_gain + high*treble
    """
    nyq = 0.5 * fs

    # Low band (Bass): 0 – 300Hz
    b_low, a_low   = butter(4, 300.0  / nyq, btype='low')
    low_band       = filtfilt(b_low,  a_low,  audio)

    # High band (Treble): 3000Hz – Nyquist
    b_high, a_high = butter(4, 3000.0 / nyq, btype='high')
    high_band      = filtfilt(b_high, a_high, audio)

    # Mid band: phần còn lại
    mid_band = audio - low_band - high_band

    # Trộn với gain
    output = (low_band  * bass_gain   +
              mid_band  * mid_gain    +
              high_band * treble_gain)

    return output


# ══════════════════════════════════════════════════════════════════
#  TẦNG 3 — PRESENCE BOOST
# ══════════════════════════════════════════════════════════════════

def _apply_presence_boost(audio, fs, center_hz=3500.0,
                           gain_db=4.0, Q=1.2):
    """
    Presence boost: tăng nhẹ vùng 2–5kHz bằng peak EQ filter.

    Tham số:
      center_hz : tần số trung tâm boost (Hz), mặc định 3500Hz
      gain_db   : mức tăng (dB), mặc định +4dB — đủ nghe rõ, không chói
      Q         : độ rộng band — Q thấp = rộng, Q cao = hẹp

    Dùng iirpeak từ scipy — bộ lọc peaking EQ chuẩn studio.
    """
    nyq    = 0.5 * fs
    w0     = center_hz / nyq
    # Giới hạn w0 trong (0, 1) để tránh lỗi
    w0     = np.clip(w0, 0.01, 0.99)
    gain_linear = 10 ** (gain_db / 20.0)

    b, a   = iirpeak(w0, Q / gain_linear)
    boosted = filtfilt(b, a, audio)

    # Trộn 60% boost + 40% gốc để tránh quá chói
    return audio * 0.4 + boosted * 0.6


# ══════════════════════════════════════════════════════════════════
#  TẦNG 4 — DE-ESSING
# ══════════════════════════════════════════════════════════════════

def _apply_deessing(audio, fs, center_hz=7500.0,
                    reduction_db=6.0, Q=2.0):
    """
    De-essing: giảm sibilance (âm "s", "sh", "ch" bị chói).

    Nguyên lý:
      Tần số sibilance thường nằm trong 5–10kHz, đỉnh phổ biến ở 7.5kHz.
      Dùng notch filter (iirnotch) để cắt giảm vùng này.

    Tham số:
      center_hz    : tần số notch (Hz), mặc định 7500Hz
      reduction_db : mức giảm (dB), mặc định -6dB
      Q            : độ hẹp của notch — Q cao = notch hẹp và chính xác hơn
    """
    nyq = 0.5 * fs

    # Chỉ áp dụng nếu sample rate đủ cao để có tần số 7.5kHz
    if center_hz >= nyq:
        center_hz = nyq * 0.85

    w0 = center_hz / nyq
    w0 = np.clip(w0, 0.01, 0.99)

    b, a    = iirnotch(w0, Q)
    notched = filtfilt(b, a, audio)

    # Trộn theo mức reduction_db
    # reduction_db = 6dB → mix ratio ~0.5
    mix = 10 ** (-reduction_db / 20.0)
    return audio * (1 - mix) + notched * mix