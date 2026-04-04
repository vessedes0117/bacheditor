import librosa
import numpy as np

def remove_silence(audio_data, top_db=30):
    """
    Hàm tự động cắt bỏ các đoạn im lặng.
    - top_db: Ngưỡng âm lượng (dB). Âm thanh nhỏ hơn mức này sẽ bị coi là im lặng và bị cắt.
    """
    print("Đang phát hiện và cắt khoảng im lặng...")
    
    # Tìm các đoạn thực sự có tiếng
    intervals = librosa.effects.split(audio_data, top_db=top_db)
    
    # Nối các đoạn có tiếng lại với nhau (bỏ qua các đoạn im lặng ở giữa)
    non_silent_audio = []
    for start, end in intervals:
        non_silent_audio.extend(audio_data[start:end])
        
    print("Cắt khoảng im lặng hoàn tất!")
    return np.array(non_silent_audio)