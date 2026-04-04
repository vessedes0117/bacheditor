import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AudioVisualizer(FigureCanvas):
    def __init__(self, parent=None):
        # Tạo khung vẽ đồ thị với nền trong suốt để khớp với giao diện
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#1E293B') 
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Chia làm 2 biểu đồ: Trên là Sóng âm, Dưới là Phổ tần số
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_spec = self.fig.add_subplot(212)
        self.setup_plot()

    def setup_plot(self):
        # Cài đặt màu sắc cho chữ và nền của đồ thị
        for ax in [self.ax_wave, self.ax_spec]:
            ax.set_facecolor('#0F172A')
            ax.tick_params(colors='#94A3B8')
            for spine in ax.spines.values():
                spine.set_color('#334155')

        self.ax_wave.set_title("Sóng Âm (Waveform)", color='#E2E8F0', fontsize=10)
        self.ax_spec.set_title("Phổ Tần Số (Spectrogram)", color='#E2E8F0', fontsize=10)
        self.fig.tight_layout()

    def plot_audio(self, audio_data, sample_rate, color='#3B82F6'):
        """Hàm nhận dữ liệu âm thanh và vẽ lên biểu đồ"""
        self.ax_wave.clear()
        self.ax_spec.clear()
        self.setup_plot()

        if audio_data is not None and len(audio_data) > 0:
            # 1. Vẽ Sóng âm (màu sắc tùy chỉnh)
            time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
            self.ax_wave.plot(time, audio_data, color=color, linewidth=0.5)

            # 2. Vẽ Phổ tần số (Cộng thêm 1e-10 để tránh lỗi chia cho 0 khi im lặng tuyệt đối)
            safe_audio = audio_data + 1e-10
            self.ax_spec.specgram(safe_audio, Fs=sample_rate, cmap='magma')

        self.draw()