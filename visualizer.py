# visualizer.py — PRO EDITION (Cập nhật Metrics Thống Kê)
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

class AudioVisualizer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('#1E293B')
        super().__init__(self.fig)
        self.setParent(parent)
        self._build_layout()

    def _build_layout(self):
        """
        Bố cục 2x2 + 1 hàng SNR & Metrics:
          [Waveform Gốc]   | [Waveform Đã Xử Lý]
          [Spectrogram Gốc]| [Spectrogram Đã Xử Lý]
          [     Chỉ số Chất lượng (Toàn chiều ngang)    ]
        """
        self.fig.clear()
        gs = GridSpec(3, 2, figure=self.fig,
                      height_ratios=[1, 1, 0.6],
                      hspace=0.55, wspace=0.35)

        self.ax_wave_before = self.fig.add_subplot(gs[0, 0])
        self.ax_wave_after  = self.fig.add_subplot(gs[0, 1])
        self.ax_spec_before = self.fig.add_subplot(gs[1, 0])
        self.ax_spec_after  = self.fig.add_subplot(gs[1, 1])
        self.ax_snr         = self.fig.add_subplot(gs[2, :])  # full width

        self._style_all_axes()

    def _style_all_axes(self):
        all_axes = [
            self.ax_wave_before, self.ax_wave_after,
            self.ax_spec_before, self.ax_spec_after,
            self.ax_snr
        ]
        for ax in all_axes:
            ax.set_facecolor('#0F172A')
            ax.tick_params(colors='#94A3B8', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        self.ax_wave_before.set_title("Sóng Âm — Gốc",      color='#94A3B8', fontsize=8)
        self.ax_wave_after.set_title("Sóng Âm — Đã Xử Lý",  color='#2DD4BF', fontsize=8)
        self.ax_spec_before.set_title("Spectrogram — Gốc",   color='#94A3B8', fontsize=8)
        self.ax_spec_after.set_title("Spectrogram — Đã Xử Lý", color='#2DD4BF', fontsize=8)
        self.ax_snr.set_title("Chỉ Số Chất Lượng (SNR & Volume)", color='#E2E8F0', fontsize=8)

    def plot_audio(self, audio_data, sample_rate, color='#475569'):
        """Vẽ bản gốc vào cột trái — cột phải vẽ mờ để làm preview."""
        self._build_layout()

        if audio_data is not None and len(audio_data) > 0:
            time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
            
            # ── Cột trái: Bản gốc ──────────────────────────────────
            self.ax_wave_before.plot(time, audio_data, color=color, linewidth=0.5)
            self.ax_wave_before.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)

            self.ax_spec_before.specgram(audio_data + 1e-10, Fs=sample_rate, cmap='magma')
            self.ax_spec_before.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)
            self.ax_spec_before.set_ylabel("Tần số (Hz)",   color='#94A3B8', fontsize=7)

            # ── Cột phải: vẽ lại bản gốc làm preview (MÀU TỐI) ────────────
            self.ax_wave_after.plot(time, audio_data, color='#1E3A4A', linewidth=0.5)
            self.ax_wave_after.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)
            self.ax_wave_after.set_title("Sóng Âm — Đã Xử Lý", color='#334155', fontsize=8)

            self.ax_spec_after.specgram(audio_data + 1e-10, Fs=sample_rate, cmap='gray')
            self.ax_spec_after.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)
            self.ax_spec_after.set_ylabel("Tần số (Hz)",   color='#94A3B8', fontsize=7)
            self.ax_spec_after.set_title("Spectrogram — Đã Xử Lý", color='#334155', fontsize=8)

            # ── SNR bar: hiển thị khung trống ────────
            self.ax_snr.barh(['Bản Gốc', 'Đã Xử Lý'], [0, 0],
                             color=['#1E293B', '#1E293B'], height=0.4)
            self.ax_snr.text(0.5, 0.5,
                             "Nhấn BẮT ĐẦU XỬ LÝ để xem các chỉ số",
                             color='#334155', ha='center', va='center',
                             fontsize=9, transform=self.ax_snr.transAxes)

            self.draw()

    def plot_comparison(self, original, processed, sample_rate):
        """
        Vẽ đầy đủ 2x2 + Các chỉ số chất lượng sau xử lý.
        Cột trái = bản gốc, cột phải = bản đã xử lý.
        """
        self._build_layout()

        if original is None or processed is None:
            return

        # Đồng bộ trục thời gian
        time_orig = np.linspace(0, len(original)  / sample_rate, num=len(original))
        time_proc = np.linspace(0, len(processed) / sample_rate, num=len(processed))

        # ── Waveform ──────────────────────────────────────────────
        self.ax_wave_before.plot(time_orig, original,  color='#475569', linewidth=0.5)
        self.ax_wave_after.plot( time_proc, processed, color='#2DD4BF', linewidth=0.5)

        for ax in [self.ax_wave_before, self.ax_wave_after]:
            ax.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)

        # ── Spectrogram ───────────────────────────────────────────
        self.ax_spec_before.specgram(original  + 1e-10, Fs=sample_rate, cmap='magma')
        self.ax_spec_after.specgram( processed + 1e-10, Fs=sample_rate, cmap='magma')

        for ax in [self.ax_spec_before, self.ax_spec_after]:
            ax.set_xlabel("Thời gian (s)", color='#94A3B8', fontsize=7)
            ax.set_ylabel("Tần số (Hz)",   color='#94A3B8', fontsize=7)

        # ── TÍNH TOÁN METRICS PRO ─────────────────────────────────
        metrics_before = compute_metrics(original, sample_rate)
        metrics_after  = compute_metrics(processed, sample_rate)
        
        snr_before  = metrics_before['SNR_est_dB']
        snr_after   = metrics_after['SNR_est_dB']
        improvement = snr_after - snr_before

        # ── Cập nhật Bar Chart ────────────────────────────────────
        labels = ['Bản Gốc', 'Đã Xử Lý']
        values = [snr_before, snr_after]
        colors = ['#475569', '#3B82F6']

        bars = self.ax_snr.barh(labels, values, color=colors, height=0.4)

        # Ghi text vào cột Bản Gốc
        self.ax_snr.text(
            bars[0].get_width() + 0.3, bars[0].get_y() + bars[0].get_height() / 2,
            f"SNR: {snr_before:.1f} dB  |  RMS: {metrics_before['RMS_dBFS']:.1f} dBFS",
            color='#94A3B8', va='center', fontsize=8
        )
        
        # Ghi text vào cột Đã Xử Lý
        self.ax_snr.text(
            bars[1].get_width() + 0.3, bars[1].get_y() + bars[1].get_height() / 2,
            f"SNR: {snr_after:.1f} dB  |  RMS: {metrics_after['RMS_dBFS']:.1f} dBFS  |  Peak: {metrics_after['Peak_dBFS']:.1f} dBFS",
            color='#F8FAFC', va='center', fontsize=8, fontweight='bold'
        )

        sign      = '+' if improvement >= 0 else ''
        color_imp = '#10B981' if improvement >= 0 else '#EF4444'
        self.ax_snr.set_title(
            f"Chỉ Số Chất Lượng — SNR Cải thiện: {sign}{improvement:.1f} dB",
            color=color_imp, fontsize=9, fontweight='bold'
        )
        self.ax_snr.set_xlabel("SNR (dB)", color='#94A3B8', fontsize=7)

        # Căn chỉnh lại trục X để text không bị cắt
        max_val = max(snr_before, snr_after)
        self.ax_snr.set_xlim(0, max_val + 20) # Cộng thêm margin để chứa được text dài

        self.draw()


# ══════════════════════════════════════════════════════════════════
#  HÀM TÍNH TOÁN CHỈ SỐ (METRICS) CHUẨN DSP
# ══════════════════════════════════════════════════════════════════

def compute_metrics(audio, sample_rate):
    """Tính RMS, Peak, SNR ước lượng thống kê (Dựa trên Percentile Energy)"""
    
    # 1. Tính RMS và Peak
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    
    # 2. Ước lượng noise floor từ 5% frame có năng lượng thấp nhất
    frame_size = int(0.05 * sample_rate) # frame 50ms
    
    # Chia audio thành các frame (có overlap 50% để chính xác hơn)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size//2)]
    
    if not frames: # Fallback nếu file quá ngắn
        return {'RMS_dBFS': 0, 'Peak_dBFS': 0, 'SNR_est_dB': 0}
        
    energies = [np.mean(f**2) for f in frames]
    
    # Phân vị: 5% thấp nhất coi là Noise, 50% (Trung vị) coi là Tín hiệu nền
    noise_floor  = np.percentile(energies, 5)
    signal_floor = np.percentile(energies, 50)
    
    # Tính SNR (Tỷ lệ tín hiệu trên nhiễu)
    snr_est = 10 * np.log10(signal_floor / (noise_floor + 1e-10))
    
    return {
        'RMS_dBFS': 20 * np.log10(rms + 1e-10),
        'Peak_dBFS': 20 * np.log10(peak + 1e-10),
        'SNR_est_dB': np.clip(snr_est, -10, 50)
    }