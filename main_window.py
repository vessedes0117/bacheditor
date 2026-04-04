import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QSlider,
                             QFileDialog, QMessageBox, QCheckBox, QSplitter)
from PyQt5.QtCore import Qt

# Import các module xử lý
import audio_input
import noise_suppression
import silence_removal
import voice_enhancement
import agc
from visualizer import AudioVisualizer
import sounddevice as sd
import soundfile as sf

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trợ Lý Âm Thanh - Pro Edition")
        self.resize(1280, 800)
        
        self.audio_data = None
        self.sample_rate = None
        self.processed_audio = None

        # Bảng màu chuẩn Tailwind Dark Mode
        self.setStyleSheet("""
            QMainWindow { background-color: #0F172A; } /* Slate 900 */
            QLabel { color: #F8FAFC; font-family: -apple-system, sans-serif; } /* Slate 50 */
            QFrame { background-color: #1E293B; border-radius: 12px; border: 1px solid #334155; } /* Slate 800 */
            QPushButton { font-family: -apple-system, sans-serif; border: none; border-radius: 8px; font-weight: bold; font-size: 14px; }
            QCheckBox { color: #E2E8F0; font-size: 14px; spacing: 10px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; background-color: #334155; border: 1px solid #475569; }
            QCheckBox::indicator:checked { background-color: #3B82F6; border: 1px solid #2563EB; }
        """)

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet("border: none; background-color: transparent;") 
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # --- PHẦN KHUNG TRÊN (Chia 2 cột: Trái và Giữa) ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        # 1. SIDEBAR TRÁI (Input & Cài đặt) - Cố định độ rộng
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(20)

        # 1.1 Mục Input
        lbl_input = QLabel("ĐẦU VÀO ÂM THANH")
        lbl_input.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold;")
        
        # --- THAO TÁC 2.1: Cập nhật Animation nút Nhập File ---
        self.btn_import = QPushButton("Nhập File Âm Thanh")
        self.btn_import.setStyleSheet("""
            QPushButton { background-color: #334155; color: white; padding: 12px; border-radius: 8px; }
            QPushButton:hover { background-color: #475569; }
            QPushButton:pressed { background-color: #1E293B; padding-top: 14px; padding-bottom: 10px; }
        """)
        
        # --- THAO TÁC 1: Cập nhật Animation nút Ghi Âm ---
        self.btn_record = QPushButton("🔴 Ghi Âm Trực Tiếp")
        self.btn_record_style = """
            QPushButton { background-color: transparent; border: 1px solid #475569; color: #F8FAFC; padding: 12px; border-radius: 8px; }
            QPushButton:hover { background-color: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; color: #EF4444; }
            QPushButton:pressed { background-color: rgba(239, 68, 68, 0.2); padding-top: 14px; padding-bottom: 10px; }
        """
        self.btn_record.setStyleSheet(self.btn_record_style)

        left_layout.addWidget(lbl_input)
        left_layout.addWidget(self.btn_import)
        left_layout.addWidget(self.btn_record)

        # Đường kẻ ngang
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #334155; border: none; max-height: 1px;")
        left_layout.addWidget(line)

        # 1.2 Mục Cài đặt tính năng
        lbl_options = QLabel("TÙY CHỌN XỬ LÝ")
        lbl_options.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_options)

        self.chk_noise = QCheckBox("Khử Tiếng Ồn Nền")
        self.chk_noise.setChecked(True)
        left_layout.addWidget(self.chk_noise)

        # Thanh trượt
        slider_layout = QHBoxLayout()
        lbl_slider_text = QLabel("Mức độ lọc:")
        lbl_slider_text.setStyleSheet("color: #94A3B8; font-size: 12px;")
        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setValue(50)
        self.slider_noise.setStyleSheet("""
            QSlider::groove:horizontal { border-radius: 3px; height: 4px; background: #334155; } 
            QSlider::handle:horizontal { background: #3B82F6; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        slider_layout.addWidget(lbl_slider_text)
        slider_layout.addWidget(self.slider_noise)
        left_layout.addLayout(slider_layout)

        self.chk_silence = QCheckBox("Cắt Khoảng Im Lặng")
        self.chk_silence.setChecked(True)
        self.chk_voice = QCheckBox("Tăng Cường Giọng Nói")
        self.chk_voice.setChecked(True)
        self.chk_agc = QCheckBox("Cân Bằng Âm Lượng (AGC)")
        self.chk_agc.setChecked(True)

        left_layout.addWidget(self.chk_silence)
        left_layout.addWidget(self.chk_voice)
        left_layout.addWidget(self.chk_agc)
        left_layout.addStretch()

        # --- THAO TÁC 2.2: Cập nhật Animation nút BẮT ĐẦU XỬ LÝ ---
        self.btn_process = QPushButton("BẮT ĐẦU XỬ LÝ")
        self.style_btn_process = """
            QPushButton { background-color: #3B82F6; color: white; padding: 16px; font-size: 15px; border-radius: 8px; }
            QPushButton:hover { background-color: #60A5FA; }
            QPushButton:pressed { background-color: #1D4ED8; padding-top: 18px; padding-bottom: 14px; }
        """
        self.btn_process.setStyleSheet(self.style_btn_process)
        left_layout.addWidget(self.btn_process)

        # 2. MAIN CENTER (Khu vực vẽ sóng âm)
        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)
        self.visualizer = AudioVisualizer(center_panel)
        center_layout.addWidget(self.visualizer)

        # Ráp 2 cột vào phần trên
        top_layout.addWidget(left_panel)
        top_layout.addWidget(center_panel, 1)

        # --- PHẦN DƯỚI (Thanh Bottom Bar) ---
        bottom_panel = QFrame()
        bottom_panel.setFixedHeight(80)
        bottom_panel.setStyleSheet("background-color: #1E293B; border-radius: 12px; border: 1px solid #334155;")
        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(24, 0, 24, 0)
        
        self.btn_play_before = QPushButton("▶ Bản Gốc")
        self.btn_play_after = QPushButton("▶ Bản Đã Xử Lý")
        
        # --- THAO TÁC 2.3: Cập nhật Animation nút Phát ---
        btn_play_style = """
            QPushButton { background-color: transparent; color: #94A3B8; padding: 10px 20px; border: 1px solid #334155; border-radius: 8px; }
            QPushButton:hover { background-color: #1E293B; color: #F8FAFC; border: 1px solid #64748B; }
            QPushButton:pressed { background-color: #0F172A; padding-top: 12px; padding-bottom: 8px; }
        """
        self.btn_play_before.setStyleSheet(btn_play_style)
        self.btn_play_after.setStyleSheet(btn_play_style)
        
        # --- THAO TÁC 2.4: Cập nhật Animation nút Xuất File ---
        self.btn_export = QPushButton("↓ Xuất File")
        self.style_btn_export = """
            QPushButton { background-color: #10B981; color: white; padding: 12px 24px; border-radius: 8px; }
            QPushButton:hover { background-color: #34D399; }
            QPushButton:pressed { background-color: #059669; padding-top: 14px; padding-bottom: 10px; }
        """
        self.btn_export.setStyleSheet(self.style_btn_export)
        
        bottom_layout.addWidget(self.btn_play_before)
        bottom_layout.addWidget(self.btn_play_after)
        bottom_layout.addStretch() 
        bottom_layout.addWidget(self.btn_export)

        # Ráp tổng thể
        main_layout.addLayout(top_layout, 1) 
        main_layout.addWidget(bottom_panel, 0)

        # ==========================================
        # KẾT NỐI SỰ KIỆN
        # ==========================================
        self.btn_import.clicked.connect(self.load_file)
        self.btn_record.clicked.connect(self.record_mic)
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_play_before.clicked.connect(self.play_before)
        self.btn_play_after.clicked.connect(self.play_after)
        self.btn_export.clicked.connect(self.export_file)

    # --- CÁC HÀM XỬ LÝ ---
    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file âm thanh", "", "Audio Files (*.wav *.mp3 *.flac)")
        if file_name:
            try:
                self.audio_data, self.sample_rate = audio_input.load_audio_file(file_name)
                self.processed_audio = None 
                self.visualizer.plot_audio(self.audio_data, self.sample_rate, color='#475569') # Màu xám cho bản gốc
                QMessageBox.information(self, "Thành công", f"Đã tải file thành công!\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể đọc file: {str(e)}")

    def record_mic(self):
        try:
            self.btn_record.setText("Đang ghi âm (5s)...")
            self.btn_record.setStyleSheet("background-color: #EF4444; color: white; padding: 12px; border: none;")
            QApplication.processEvents()
            
            self.audio_data, self.sample_rate = audio_input.record_audio(duration=5)
            self.processed_audio = None 
            self.visualizer.plot_audio(self.audio_data, self.sample_rate, color='#475569')
            
            self.btn_record.setText("🔴 Ghi Âm Trực Tiếp")
            self.btn_record.setStyleSheet(self.btn_record_style)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể ghi âm: {str(e)}")
            self.btn_record.setText("🔴 Ghi Âm Trực Tiếp")
            self.btn_record.setStyleSheet(self.btn_record_style)

    def process_audio(self):
        if self.audio_data is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập file hoặc ghi âm trước!")
            return

        try:
            self.btn_process.setText("ĐANG XỬ LÝ...")
            self.btn_process.setStyleSheet("background-color: #2563EB; color: white; padding: 16px; font-size: 15px;")
            QApplication.processEvents()

            current_audio = self.audio_data.copy()

            if self.chk_noise.isChecked():
                noise_level = self.slider_noise.value() / 100.0
                current_audio = noise_suppression.reduce_noise(current_audio, self.sample_rate, noise_level)

            if self.chk_silence.isChecked():
                current_audio = silence_removal.remove_silence(current_audio)

            if self.chk_voice.isChecked():
                current_audio = voice_enhancement.enhance_voice(current_audio, self.sample_rate)

            if self.chk_agc.isChecked():
                current_audio = agc.apply_agc(current_audio)

            self.processed_audio = current_audio
            self.visualizer.plot_audio(self.processed_audio, self.sample_rate, color='#2DD4BF') # Màu Teal sáng cho bản đã xử lý

            # --- THAO TÁC 3: Trả lại bộ CSS Animation và cập nhật nút Play After ---
            self.btn_process.setText("BẮT ĐẦU XỬ LÝ")
            self.btn_process.setStyleSheet(self.style_btn_process) # Trả lại bộ CSS Animation
            self.btn_play_after.setStyleSheet("""
                QPushButton { background-color: rgba(59, 130, 246, 0.1); color: #3B82F6; padding: 10px 20px; border: 1px solid #3B82F6; border-radius: 8px; }
                QPushButton:hover { background-color: rgba(59, 130, 246, 0.2); }
                QPushButton:pressed { padding-top: 12px; padding-bottom: 8px; }
            """)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Có lỗi khi xử lý: {str(e)}")
            self.btn_process.setText("BẮT ĐẦU XỬ LÝ")
            self.btn_process.setStyleSheet(self.style_btn_process)

    def play_before(self):
        if self.audio_data is not None:
            sd.play(self.audio_data, self.sample_rate)

    def play_after(self):
        if self.processed_audio is not None:
            sd.play(self.processed_audio, self.sample_rate)
        else:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có âm thanh đã xử lý!")

    def export_file(self):
        if self.processed_audio is None:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có âm thanh để xuất!")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "Lưu file", "Audio_Da_Xu_Ly.wav", "Audio Files (*.wav)")
        if file_name:
            try:
                sf.write(file_name, self.processed_audio, self.sample_rate)
                QMessageBox.information(self, "Thành công", f"Đã lưu file tại:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file: {str(e)}")

# Đảm bảo hàm init_ui được gọi trong __init__
MainWindow.init_ui = MainWindow.init_ui # type: ignore
def start_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.init_ui()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    start_app()