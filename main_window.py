# main_window.py
import sys
import time as _time
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame, QSlider,
                             QFileDialog, QMessageBox, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sounddevice as sd
import soundfile as sf

import audio_input
import noise_suppression
import silence_removal
import voice_enhancement
import agc
from visualizer import AudioVisualizer


# ══════════════════════════════════════════════════════════════════
#  RECORDING THREAD
# ══════════════════════════════════════════════════════════════════
class RecordingThread(QThread):
    finished = pyqtSignal(np.ndarray, int)
    tick     = pyqtSignal(int)

    def __init__(self, samplerate=44100):
        super().__init__()
        self.samplerate  = samplerate
        self._is_running = False
        self._chunks     = []

    def run(self):
        self._is_running = True
        self._chunks     = []
        start_time       = _time.time()

        with sd.InputStream(samplerate=self.samplerate,
                            channels=1, dtype='float32') as stream:
            while self._is_running:
                chunk, _ = stream.read(self.samplerate // 10)
                self._chunks.append(chunk.flatten())
                elapsed = int(_time.time() - start_time)
                self.tick.emit(elapsed)
                _time.sleep(0.05)

        if self._chunks:
            audio = np.concatenate(self._chunks)
            self.finished.emit(audio, self.samplerate)

    def stop(self):
        self._is_running = False


# ══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trợ Lý Âm Thanh - Pro Edition")
        self.resize(1280, 800)

        self.audio_data       = None
        self.sample_rate      = None
        self.processed_audio  = None
        self.recording_thread = None

        self._playing         = None   
        self._paused          = False
        self._pause_position  = 0      
        self._play_start_time = None  

        self.setStyleSheet("""
            QMainWindow { background-color: #0F172A; }
            QLabel { color: #F8FAFC; font-family: -apple-system, sans-serif; }
            QFrame { background-color: #1E293B; border-radius: 12px; border: 1px solid #334155; }
            QPushButton { font-family: -apple-system, sans-serif; border: none; border-radius: 8px; font-weight: bold; font-size: 14px; }
            QCheckBox { color: #E2E8F0; font-size: 14px; spacing: 10px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; background-color: #334155; border: 1px solid #475569; }
            QCheckBox::indicator:checked { background-color: #3B82F6; border: 1px solid #2563EB; }
        """)

        self.init_ui()

    def _make_slider(self, label, tooltip, attr_name, val_attr,
                     min_val, max_val, default, unit=''):
        """Helper tạo slider có label + giá trị hiển thị — tái sử dụng nhiều nơi."""
        layout   = QHBoxLayout()
        lbl      = QLabel(label)
        lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        lbl.setToolTip(tooltip)
        lbl.setFixedWidth(110)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setStyleSheet("""
            QSlider::groove:horizontal { border-radius: 3px; height: 4px; background: #334155; }
            QSlider::handle:horizontal { background: #3B82F6; width: 12px;
                                         margin: -4px 0; border-radius: 6px; }
        """)

        val_lbl = QLabel(f"{default}{unit}")
        val_lbl.setStyleSheet("color: #3B82F6; font-size: 11px; min-width: 40px;")
        slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v}{unit}"))

        setattr(self, attr_name, slider)
        setattr(self, val_attr,  val_lbl)

        layout.addWidget(lbl)
        layout.addWidget(slider)
        layout.addWidget(val_lbl)
        return layout

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("border: none; background-color: transparent;")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        # ── SIDEBAR TRÁI ──────────────────────────────────────────
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(14)

        lbl_input = QLabel("ĐẦU VÀO ÂM THANH")
        lbl_input.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold;")

        self.btn_import = QPushButton("Nhập File Âm Thanh")
        self.btn_import.setStyleSheet("""
            QPushButton { background-color: #334155; color: white; padding: 12px; border-radius: 8px; }
            QPushButton:hover { background-color: #475569; }
            QPushButton:pressed { background-color: #1E293B; padding-top: 14px; padding-bottom: 10px; }
        """)

        self.btn_record_style = """
            QPushButton { background-color: transparent; border: 1px solid #475569; color: #F8FAFC; padding: 12px; border-radius: 8px; }
            QPushButton:hover { background-color: rgba(239,68,68,0.1); border: 1px solid #EF4444; color: #EF4444; }
            QPushButton:pressed { background-color: rgba(239,68,68,0.2); padding-top: 14px; padding-bottom: 10px; }
        """
        self.btn_record = QPushButton("🔴 Ghi Âm Trực Tiếp")
        self.btn_record.setStyleSheet(self.btn_record_style)

        left_layout.addWidget(lbl_input)
        left_layout.addWidget(self.btn_import)
        left_layout.addWidget(self.btn_record)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #334155; border: none; max-height: 1px;")
        left_layout.addWidget(line1)

        lbl_info = QLabel("THÔNG TIN FILE")
        lbl_info.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_info)

        info_style = "color: #64748B; font-size: 12px; padding: 2px 0px;"
        self.lbl_duration   = QLabel("⏱  Thời lượng  :  —")
        self.lbl_samplerate = QLabel("📶  Sample Rate :  —")
        self.lbl_samples    = QLabel("🔢  Số mẫu      :  —")
        for lbl in [self.lbl_duration, self.lbl_samplerate, self.lbl_samples]:
            lbl.setStyleSheet(info_style)
            left_layout.addWidget(lbl)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #334155; border: none; max-height: 1px;")
        left_layout.addWidget(line2)

        # ── CẬP NHẬT: Tùy chọn xử lý GỌN GÀNG ───────────────────────────
        lbl_options = QLabel("TÙY CHỌN XỬ LÝ")
        lbl_options.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_options)

        # Khử tiếng ồn — 1 slider
        self.chk_noise = QCheckBox("Khử Tiếng Ồn Nền")
        self.chk_noise.setChecked(True)
        left_layout.addWidget(self.chk_noise)
        left_layout.addLayout(self._make_slider(
            label='Mức độ lọc:', tooltip='Càng cao càng lọc mạnh',
            attr_name='slider_noise', val_attr='lbl_noise_value',
            min_val=0, max_val=100, default=50, unit='%'
        ))

        # Cắt im lặng — 1 slider
        self.chk_silence = QCheckBox("Cắt Khoảng Im Lặng")
        self.chk_silence.setChecked(True)
        left_layout.addWidget(self.chk_silence)
        left_layout.addLayout(self._make_slider(
            label='Ngưỡng im lặng:', tooltip='Càng cao càng cắt nhiều',
            attr_name='slider_top_db', val_attr='lbl_top_db_val',
            min_val=10, max_val=60, default=35, unit='dB'
        ))

        self.btn_preview_silence = QPushButton("🔍 Xem Trước Kết Quả Cắt")
        self.btn_preview_silence.setStyleSheet("""
            QPushButton { background-color: transparent; color: #94A3B8; padding: 7px;
                          border: 1px solid #334155; border-radius: 6px; font-size: 12px; }
            QPushButton:hover { background-color: #334155; color: #F8FAFC; }
        """)
        self.btn_preview_silence.clicked.connect(self.preview_silence)
        left_layout.addWidget(self.btn_preview_silence)

        self.lbl_silence_preview = QLabel("")
        self.lbl_silence_preview.setStyleSheet("color: #64748B; font-size: 11px;")
        self.lbl_silence_preview.setWordWrap(True)
        left_layout.addWidget(self.lbl_silence_preview)

        # Tăng cường giọng nói — 1 slider "Độ rõ giọng"
        self.chk_voice = QCheckBox("Tăng Cường Giọng Nói")
        self.chk_voice.setChecked(True)
        left_layout.addWidget(self.chk_voice)
        left_layout.addLayout(self._make_slider(
            label='Độ rõ giọng:', tooltip='Tăng để giọng sắc nét, rõ hơn',
            attr_name='slider_clarity', val_attr='lbl_clarity_val',
            min_val=0, max_val=200, default=100, unit='%'
        ))

        # AGC
        self.chk_agc = QCheckBox("Cân Bằng Âm Lượng (AGC)")
        self.chk_agc.setChecked(True)
        left_layout.addWidget(self.chk_agc)
        
        left_layout.addStretch()

        self.style_btn_process = """
            QPushButton { background-color: #3B82F6; color: white; padding: 16px; font-size: 15px; border-radius: 8px; }
            QPushButton:hover { background-color: #60A5FA; }
            QPushButton:pressed { background-color: #1D4ED8; padding-top: 18px; padding-bottom: 14px; }
        """
        self.btn_process = QPushButton("BẮT ĐẦU XỬ LÝ")
        self.btn_process.setStyleSheet(self.style_btn_process)
        left_layout.addWidget(self.btn_process)

        # ── CENTER: Visualizer ────────────────────────────────────
        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)
        self.visualizer = AudioVisualizer(center_panel)
        center_layout.addWidget(self.visualizer)

        top_layout.addWidget(left_panel)
        top_layout.addWidget(center_panel, 1)

        # ── BOTTOM BAR ────────────────────────────────────────────
        bottom_panel = QFrame()
        bottom_panel.setFixedHeight(80)
        bottom_panel.setStyleSheet(
            "background-color: #1E293B; border-radius: 12px; border: 1px solid #334155;")
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(24, 0, 24, 0)
        bottom_layout.setSpacing(10)

        btn_play_style = """
            QPushButton { background-color: transparent; color: #94A3B8; padding: 10px 20px; border: 1px solid #334155; border-radius: 8px; }
            QPushButton:hover { background-color: #1E293B; color: #F8FAFC; border: 1px solid #64748B; }
            QPushButton:pressed { background-color: #0F172A; padding-top: 12px; padding-bottom: 8px; }
        """
        self.btn_play_before = QPushButton("▶ Bản Gốc")
        self.btn_play_after  = QPushButton("▶ Bản Đã Xử Lý")
        self.btn_play_before.setStyleSheet(btn_play_style)
        self.btn_play_after.setStyleSheet(btn_play_style)

        self.style_btn_pause_idle = """
            QPushButton { background-color: transparent; color: #475569; padding: 10px 20px;
                          border: 1px solid #1E293B; border-radius: 8px; }
        """
        self.style_btn_pause_active = """
            QPushButton { background-color: transparent; color: #94A3B8; padding: 10px 20px;
                          border: 1px solid #334155; border-radius: 8px; }
            QPushButton:hover { background-color: rgba(251,191,36,0.1); color: #FBBF24; border: 1px solid #FBBF24; }
        """
        self.style_btn_resume = """
            QPushButton { background-color: rgba(59,130,246,0.15); color: #3B82F6;
                          padding: 10px 20px; border: 1px solid #3B82F6; border-radius: 8px; }
            QPushButton:hover { background-color: rgba(59,130,246,0.25); }
        """
        self.btn_pause = QPushButton("⏸ Tạm Dừng")
        self.btn_pause.setStyleSheet(self.style_btn_pause_idle)
        self.btn_pause.setEnabled(False)

        self.style_btn_export = """
            QPushButton { background-color: #10B981; color: white; padding: 12px 24px; border-radius: 8px; }
            QPushButton:hover { background-color: #34D399; }
            QPushButton:pressed { background-color: #059669; padding-top: 14px; padding-bottom: 10px; }
        """
        self.btn_export = QPushButton("↓ Xuất File")
        self.btn_export.setStyleSheet(self.style_btn_export)

        bottom_layout.addWidget(self.btn_play_before)
        bottom_layout.addWidget(self.btn_play_after)
        bottom_layout.addWidget(self.btn_pause)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_export)

        main_layout.addLayout(top_layout, 1)
        main_layout.addWidget(bottom_panel, 0)

        # ── KẾT NỐI SỰ KIỆN ──────────────────────────────────────
        self.btn_import.clicked.connect(self.load_file)
        self.btn_record.clicked.connect(self.record_mic)
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_play_before.clicked.connect(self.play_before)
        self.btn_play_after.clicked.connect(self.play_after)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_export.clicked.connect(self.export_file)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file", "", "Audio (*.wav *.mp3 *.flac)")
        if file_name:
            try:
                self.audio_data, self.sample_rate = audio_input.load_audio_file(file_name)
                self.processed_audio = None
                self.lbl_silence_preview.setText("")
                self._update_file_info()
                self.visualizer.plot_audio(self.audio_data, self.sample_rate)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def _update_file_info(self):
        if self.audio_data is None: return
        duration = len(self.audio_data) / self.sample_rate
        self.lbl_duration.setText(f"⏱  Thời lượng  :  {duration:.2f}s")
        self.lbl_samplerate.setText(f"📶  Sample Rate :  {self.sample_rate} Hz")
        self.lbl_samples.setText(f"🔢  Số mẫu      :  {len(self.audio_data):,}")

    def preview_silence(self):
        if self.audio_data is None:
            self.lbl_silence_preview.setText("⚠ Chưa có file âm thanh.")
            return
        info = silence_removal.get_silence_preview(
            self.audio_data, top_db=self.slider_top_db.value(), sample_rate=self.sample_rate
        )
        self.lbl_silence_preview.setText(
            f"📊 {info['n_segments']} đoạn có tiếng\n✂ Cắt: {info['removed_duration']:.2f}s → Còn: {info['kept_duration']:.2f}s"
        )
        self.lbl_silence_preview.setStyleSheet("color: #10B981; font-size: 11px;")

    def record_mic(self):
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            return
        self.recording_thread = RecordingThread()
        self.recording_thread.tick.connect(lambda s: self.btn_record.setText(f"⏹ Dừng ({s}s)"))
        self.recording_thread.finished.connect(self._on_record_finished)
        self.recording_thread.start()
        self.btn_record.setStyleSheet("background-color: #EF4444; color: white; padding: 12px;")

    def _on_record_finished(self, audio, sr):
        self.audio_data, self.sample_rate = audio, sr
        self.processed_audio = None
        self._update_file_info()
        self.visualizer.plot_audio(audio, sr)
        self.btn_record.setText("🔴 Ghi Âm Trực Tiếp")
        self.btn_record.setStyleSheet(self.btn_record_style)

    # --- CẬP NHẬT: LOGIC XỬ LÝ MỚI ─────────────────────────────
    def process_audio(self):
        if self.audio_data is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập file hoặc ghi âm trước!")
            return
        try:
            self.btn_process.setText("ĐANG XỬ LÝ...")
            self.btn_process.setStyleSheet(
                "background-color: #2563EB; color: white; padding: 16px; font-size: 15px;")
            QApplication.processEvents()

            current_audio = self.audio_data.copy()

            if self.chk_noise.isChecked():
                noise_level   = self.slider_noise.value() / 100.0
                current_audio = noise_suppression.reduce_noise(
                    current_audio, self.sample_rate,
                    noise_level=noise_level, method='spectral')

            if self.chk_silence.isChecked():
                current_audio = silence_removal.remove_silence(
                    current_audio,
                    top_db      = self.slider_top_db.value(),
                    sample_rate = self.sample_rate)

            if self.chk_voice.isChecked():
                clarity = self.slider_clarity.value() / 100.0
                # Gộp clarity vào mid + treble gain, bass cố định 1.0
                current_audio = voice_enhancement.enhance_voice(
                    current_audio, self.sample_rate,
                    bass_gain      = 1.0,
                    mid_gain       = clarity,
                    treble_gain    = min(clarity * 1.1, 2.0),
                    presence_boost = clarity > 0.8,   # auto bật khi clarity cao
                    de_essing      = True)

            if self.chk_agc.isChecked():
                current_audio = agc.apply_agc(
                    current_audio, sample_rate=self.sample_rate)

            self.processed_audio = current_audio
            self.visualizer.plot_comparison(
                self.audio_data, self.processed_audio, self.sample_rate)

            self.btn_process.setText("BẮT ĐẦU XỬ LÝ")
            self.btn_process.setStyleSheet(self.style_btn_process)

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Có lỗi khi xử lý: {str(e)}")
            self.btn_process.setText("BẮT ĐẦU XỬ LÝ")
            self.btn_process.setStyleSheet(self.style_btn_process)

    def play_before(self):
        if self.audio_data is None: return
        self._start_playback('before')

    def play_after(self):
        if self.processed_audio is None: return
        self._start_playback('after')

    def _get_audio(self):
        return self.audio_data if self._playing == 'before' else self.processed_audio

    def _start_playback(self, source, from_sample=0):
        sd.stop()
        self._paused      = False
        self._playing     = source
        audio = self.audio_data if source == 'before' else self.processed_audio
        audio_segment = audio[from_sample:]
        if len(audio_segment) == 0:
            self._reset_playback_ui()
            return
        self._pause_position  = from_sample
        self._play_start_time = _time.time()
        sd.play(audio_segment, self.sample_rate)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Tạm Dừng")
        self.btn_pause.setStyleSheet(self.style_btn_pause_active)

    def toggle_pause(self):
        if self._playing is None: return
        if not self._paused:
            elapsed_samples  = int((_time.time() - self._play_start_time) * self.sample_rate)
            self._pause_position = min(self._pause_position + elapsed_samples, len(self._get_audio()) - 1)
            sd.stop()
            self._paused = True
            self.btn_pause.setText("▶ Tiếp Tục")
            self.btn_pause.setStyleSheet(self.style_btn_resume)
        else:
            self._start_playback(self._playing, from_sample=self._pause_position)

    def _reset_playback_ui(self):
        self._playing = None
        self._paused = False
        self.btn_pause.setText("⏸ Tạm Dừng")
        self.btn_pause.setStyleSheet(self.style_btn_pause_idle)
        self.btn_pause.setEnabled(False)

    def export_file(self):
        if self.processed_audio is None: return
        name, _ = QFileDialog.getSaveFileName(self, "Lưu file", "output.wav", "WAV (*.wav)")
        if name:
            sf.write(name, self.processed_audio, self.sample_rate)
            QMessageBox.information(self, "Lưu file", "Thành công!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())