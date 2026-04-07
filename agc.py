# agc.py
import numpy as np

def apply_agc(audio_data, target_peak=0.9,
              attack_ms=10, release_ms=100, sample_rate=44100):
    """
    Auto Gain Control (AGC) thực sự — có attack/release time.

    Vấn đề bản cũ (Peak Normalization đơn giản):
      Chỉ tìm đỉnh lớn nhất rồi nhân toàn bộ file với 1 hệ số cố định.
      → Không phản ứng theo thời gian, gây "pumping artifact"
        (âm lượng tăng đột ngột khi đoạn ồn kết thúc).

    Giải pháp — AGC có attack/release:
      - Attack : tốc độ GIẢM gain khi âm lượng tăng đột ngột (nhanh ~10ms)
      - Release: tốc độ TĂNG gain khi âm lượng giảm xuống     (chậm ~100ms)
      → Chuyển tiếp mượt mà, không bị pumping.

    Tham số:
      - target_peak : biên độ đích (0.0–1.0), mặc định 0.9
      - attack_ms   : thời gian attack  (ms), mặc định 10ms
      - release_ms  : thời gian release (ms), mặc định 100ms
      - sample_rate : tần số lấy mẫu (Hz), mặc định 44100
    """
    print("Đang cân bằng âm lượng (AGC có attack/release)...")

    if np.max(np.abs(audio_data)) == 0:
        return audio_data

    # ── Bước 1: Tính hệ số attack/release theo số mẫu ────────────
    # Công thức: coeff = exp(-1 / (time_ms * sample_rate / 1000))
    # → Gần 1.0 = chậm phản ứng, gần 0.0 = nhanh phản ứng
    attack_coeff  = np.exp(-1.0 / (attack_ms  * sample_rate / 1000))
    release_coeff = np.exp(-1.0 / (release_ms * sample_rate / 1000))

    # ── Bước 2: Tính envelope (đường bao biên độ) ────────────────
    # Envelope theo dõi mức năng lượng tức thời của tín hiệu
    envelope = np.zeros(len(audio_data))
    envelope[0] = np.abs(audio_data[0])

    for i in range(1, len(audio_data)):
        current_abs = np.abs(audio_data[i])
        if current_abs > envelope[i - 1]:
            # Biên độ tăng → dùng attack (phản ứng nhanh)
            envelope[i] = attack_coeff  * envelope[i - 1] + (1 - attack_coeff)  * current_abs
        else:
            # Biên độ giảm → dùng release (phản ứng chậm)
            envelope[i] = release_coeff * envelope[i - 1] + (1 - release_coeff) * current_abs

    # ── Bước 3: Tính gain động theo envelope ─────────────────────
    # Gain = target / envelope, nhưng giới hạn trong [0.1, 10.0]
    # tránh khuếch đại quá mức đoạn im lặng tuyệt đối
    safe_envelope = np.maximum(envelope, 1e-6)
    gain          = np.clip(target_peak / safe_envelope, 0.1, 10.0)

    # ── Bước 4: Áp dụng gain động vào tín hiệu ───────────────────
    output = audio_data * gain

    # ── Bước 5: Hard limiter — đảm bảo không vượt quá ±1.0 ──────
    output = np.clip(output, -1.0, 1.0)

    print("Cân bằng âm lượng hoàn tất!")
    return output.astype(np.float32)