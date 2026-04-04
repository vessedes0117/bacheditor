import numpy as np
from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    # Thiết lập bộ lọc âm thanh
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def enhance_voice(audio_data, sample_rate):
    """
    Hàm làm rõ giọng nói bằng cách cắt bỏ các âm trầm rền (dưới 80Hz).
    """
    print("Đang tăng cường độ trong trẻo của giọng nói...")
    
    # Tạo bộ lọc High-pass cắt tần số dưới 80Hz
    b, a = butter_highpass(80.0, sample_rate, order=5)
    
    # Áp dụng bộ lọc vào file âm thanh
    enhanced_audio = lfilter(b, a, audio_data)
    
    # Bù lại âm lượng cho bằng với bản gốc để không bị nghe nhỏ đi
    if np.max(np.abs(enhanced_audio)) > 0:
        enhanced_audio = enhanced_audio * (np.max(np.abs(audio_data)) / np.max(np.abs(enhanced_audio)))
    
    print("Tăng cường giọng nói hoàn tất!")
    return enhanced_audio