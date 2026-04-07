# audio_input.py
import soundfile as sf
import sounddevice as sd
import numpy as np

def load_audio_file(file_path):
    """Đọc file âm thanh, tự động chuyển stereo → mono."""
    data, samplerate = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data, samplerate

def record_audio_stream(samplerate=44100):
    """
    Ghi âm theo kiểu stream (không giới hạn thời gian).
    Trả về InputStream — caller tự quản lý start/stop.
    Dữ liệu được đẩy vào callback mỗi khi có buffer mới.
    """
    return sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )