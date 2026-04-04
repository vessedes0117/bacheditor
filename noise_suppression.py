import noisereduce as nr

def reduce_noise(audio_data, sample_rate, noise_level=1.0):
    """
    Hàm lọc tiếng ồn nền.
    - audio_data: dữ liệu âm thanh đầu vào
    - sample_rate: tần số lấy mẫu
    - noise_level: mức độ lọc (từ 0.0 đến 1.0)
    """
    print(f"Đang lọc ồn với mức độ: {noise_level}...")
    
    # Sử dụng thư viện noisereduce để khử ồn
    processed_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=noise_level)
    
    print("Lọc ồn hoàn tất!")
    return processed_audio