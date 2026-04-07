# visualizer.py — PHIÊN BẢN HOÀN CHỈNH
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
        Bố cục 2x2 + 1 hàng SNR:
          [Waveform Gốc]   | [Waveform Đã Xử Lý]
          [Spectrogram Gốc]| [Spectrogram Đã Xử Lý]
          [     SNR Bar (toàn chiều ngang)          ]
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
        self.ax_snr.set_title("Chỉ Số Chất Lượng (SNR)",     color='#E2E8F0', fontsize=8)

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
                             "Nhấn BẮT ĐẦU XỬ LÝ để xem chỉ số SNR",
                             color='#334155', ha='center', va='center',
                             fontsize=9, transform=self.ax_snr.transAxes)

        self.draw()

    def plot_comparison(self, original, processed, sample_rate):
        """
        Vẽ đầy đủ 2x2 + SNR sau khi xử lý xong.
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

        # ── SNR Bar ───────────────────────────────────────────────
        snr_before  = _compute_snr(original)
        snr_after   = _compute_snr(processed)
        improvement = snr_after - snr_before

        labels = ['Bản Gốc', 'Đã Xử Lý']
        values = [snr_before, snr_after]
        colors = ['#475569', '#3B82F6']

        bars = self.ax_snr.barh(labels, values, color=colors, height=0.4)

        for bar, val in zip(bars, values):
            self.ax_snr.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} dB",
                color='#F8FAFC', va='center', fontsize=8
            )

        sign      = '+' if improvement >= 0 else ''
        color_imp = '#10B981' if improvement >= 0 else '#EF4444'
        self.ax_snr.set_title(
            f"Chỉ Số Chất Lượng (SNR) — Cải thiện: {sign}{improvement:.1f} dB",
            color=color_imp, fontsize=8
        )
        self.ax_snr.set_xlabel("dB", color='#94A3B8', fontsize=7)

        self.draw()


# ══════════════════════════════════════════════════════════════════
#  HÀM TÍNH SNR
# ══════════════════════════════════════════════════════════════════

def _compute_snr(audio, noise_duration_ratio=0.1):
    """
    Ước lượng SNR (dB).
    """
    n_noise      = max(1, int(len(audio) * noise_duration_ratio))
    noise        = audio[:n_noise]
    signal       = audio[n_noise:]
    power_noise  = np.mean(noise  ** 2) + 1e-10
    power_signal = np.mean(signal ** 2) + 1e-10
    snr_db       = 10 * np.log10(power_signal / power_noise)
    return float(np.clip(snr_db, -20, 60))