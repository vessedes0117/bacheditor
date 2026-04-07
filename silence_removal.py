# silence_removal.py
import librosa
import numpy as np

def get_silence_preview(audio_data, top_db=30, sample_rate=44100):
    """Phân tích trước — không chỉnh sửa audio."""
    intervals = librosa.effects.split(audio_data, top_db=top_db)

    total_duration   = len(audio_data) / sample_rate
    kept_samples     = sum(e - s for s, e in intervals)
    kept_duration    = kept_samples / sample_rate
    removed_duration = total_duration - kept_duration

    return {
        'n_segments'      : len(intervals),
        'total_duration'  : total_duration,
        'kept_duration'   : kept_duration,
        'removed_duration': removed_duration,
    }


def remove_silence(audio_data, top_db=30, sample_rate=44100):
    """
    Cắt im lặng với 1 tham số chính: top_db (ngưỡng im lặng).
    fade_ms=10 và min_silence=100ms hardcode — giá trị tối ưu cho hầu hết trường hợp.
    """
    print(f"Đang cắt im lặng [ngưỡng={top_db}dB]...")

    FADE_MS     = 10   # hardcode — đủ mượt, không ảnh hưởng nội dung
    MIN_MS      = 100  # hardcode — lọc tiếng động nhỏ lẻ

    intervals = librosa.effects.split(audio_data, top_db=top_db)

    if len(intervals) == 0:
        print("Không tìm thấy đoạn có tiếng, giữ nguyên.")
        return audio_data

    min_samples  = int(MIN_MS  * sample_rate / 1000)
    fade_samples = int(FADE_MS * sample_rate / 1000)
    intervals    = [(s, e) for s, e in intervals if (e - s) >= min_samples]

    if len(intervals) == 0:
        return audio_data

    fade_in  = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    segments = []
    for start, end in intervals:
        segment = audio_data[start:end].copy()
        if len(segment) >= fade_samples * 2:
            segment[:fade_samples]  *= fade_in
            segment[-fade_samples:] *= fade_out
        segments.append(segment)

    result = np.concatenate(segments)
    print(f"Hoàn tất! {len(intervals)} đoạn — {len(result)/sample_rate:.2f}s / {len(audio_data)/sample_rate:.2f}s")
    return result.astype(np.float32)