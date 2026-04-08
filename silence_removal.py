# silence_removal.py
import librosa
import numpy as np

def get_silence_preview(audio_data, top_db=30, sample_rate=44100):
    """Phân tích trước để hiển thị UI — Dùng logic mới để chính xác hơn."""
    intervals = _detect_speech_segments(audio_data, top_db=top_db, sample_rate=sample_rate)
    
    total_duration = len(audio_data) / sample_rate
    kept_samples = sum(e - s for s, e in intervals)
    kept_duration = kept_samples / sample_rate
    removed_duration = total_duration - kept_duration
    
    return {
        'n_segments': len(intervals),
        'total_duration': total_duration,
        'kept_duration': kept_duration,
        'removed_duration': removed_duration,
    }

def remove_silence(audio_data, top_db=30, sample_rate=44100):
    """
    Cắt im lặng với cơ chế:
    1. Phát hiện tiếng nói.
    2. Thêm vùng đệm (Margin) 50ms để chống cụt chữ.
    3. Gộp (Merge) các đoạn gần nhau để tránh cắt rời rạc.
    """
    print(f"Đang cắt im lặng [ngưỡng={top_db}dB]...")
    
    # FADE SETTINGS
    FADE_MS = 10   # Mềm hóa điểm cắt
    
    # 1. Detect speech segments (Đã bao gồm Margin & Merge logic)
    intervals = _detect_speech_segments(audio_data, top_db=top_db, sample_rate=sample_rate)
    
    if len(intervals) == 0:
        print("⚠ Không tìm thấy đoạn có tiếng, giữ nguyên.")
        return audio_data

    # 2. Xử lý Fade In/Out cho từng đoạn
    fade_samples = int(FADE_MS * sample_rate / 1000)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    
    segments = []
    for start, end in intervals:
        segment = audio_data[start:end].copy()
        
        # Chỉ apply fade nếu đoạn đủ dài
        if len(segment) >= fade_samples * 2:
            segment[:fade_samples] *= fade_in
            segment[-fade_samples:] *= fade_out
        
        segments.append(segment)
    
    # 3. Nối lại thành 1 file
    result = np.concatenate(segments)
    
    original_dur = len(audio_data) / sample_rate
    new_dur = len(result) / sample_rate
    print(f"✅ Hoàn tất! {len(intervals)} đoạn — {new_dur:.2f}s / {original_dur:.2f}s "
          f"(giảm {100*(1-new_dur/original_dur):.1f}%)")
    
    return result.astype(np.float32)

def _detect_speech_segments(audio_data, top_db, sample_rate):
    """
    Core logic: Detect -> Add Margin -> Merge
    """
    # Bước 1: Dùng librosa detect cơ bản
    raw_intervals = librosa.effects.split(audio_data, top_db=top_db)
    
    # Bước 2: Cấu hình Margin (Vùng đệm an toàn)
    # 50ms (0.05s) là đủ để giữ lại các phụ âm cuối (s, t, p) và hơi thở
    MARGIN_MS = 50 
    margin_samples = int(MARGIN_MS * sample_rate / 1000)
    
    # Bước 3: Thêm Margin
    padded_intervals = []
    for start, end in raw_intervals:
        # Lùi lại và tiến thêm 50ms
        s = max(0, start - margin_samples)
        e = min(len(audio_data), end + margin_samples)
        padded_intervals.append([s, e])
    
    # Bước 4: Merge (Gộp) các đoạn chồng lấn
    # Nếu đoạn sau cách đoạn trước < 200ms, gộp lại thành 1 đoạn lớn
    # -> Giúp câu nói liền mạch, không bị ngắt quãng vô lý
    MERGE_GAP_MS = 200
    gap_samples = int(MERGE_GAP_MS * sample_rate / 1000)
    
    merged = []
    if len(padded_intervals) > 0:
        current_start, current_end = padded_intervals[0]
        
        for next_start, next_end in padded_intervals[1:]:
            # Nếu khoảng cách giữa 2 đoạn nhỏ hơn 200ms -> Gộp
            if next_start < current_end + gap_samples:
                current_end = max(current_end, next_end) # Mở rộng đoạn hiện tại
            else:
                # Nếu cách xa -> Đóng đoạn cũ, mở đoạn mới
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        # Đừng quên đoạn cuối cùng
        merged.append((current_start, current_end))
    
    return merged