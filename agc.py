import numpy as np

def apply_agc(audio_data, target_peak=0.9):
    """
    Hàm Cân bằng âm lượng (Auto Gain Control).
    Khuếch đại âm thanh lên mức chuẩn (target_peak) để nghe rõ ràng hơn.
    """
    print("Đang tự động cân bằng âm lượng...")
    
    # Tìm mức âm lượng lớn nhất hiện tại trong file
    max_peak = np.max(np.abs(audio_data))
    
    # Nếu file không có tiếng gì thì giữ nguyên để tránh lỗi
    if max_peak == 0:
        return audio_data 
        
    # Tính toán mức độ cần tăng/giảm và áp dụng vào âm thanh
    gain = target_peak / max_peak
    balanced_audio = audio_data * gain
    
    print("Cân bằng âm lượng hoàn tất!")
    return balanced_audio