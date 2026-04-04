import soundfile as sf
import sounddevice as sd
import numpy as np

def load_audio_file(file_path):
    # Đọc file âm thanh (hỗ trợ wav, flac...)
    data, samplerate = sf.read(file_path)
    
    # Nếu âm thanh là stereo (2 kênh), chuyển về mono (1 kênh) để dễ xử lý
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    return data, samplerate

def record_audio(duration=5, samplerate=44100):
    # Ghi âm từ microphone mặc định của máy (mặc định ghi 5 giây)
    print("Đang ghi âm...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait() # Đợi cho đến khi ghi âm xong
    print("Ghi âm hoàn tất!")
    
    return recording.flatten(), samplerate