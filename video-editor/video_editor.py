"""
Video Editor with Highlighted Subtitles
========================================

A desktop application that:
1. Extracts audio from uploaded video files
2. Generates word-level synchronized subtitles
3. Renders videos with highlighted subtitles (like YouTube Shorts/Instagram Reels)
4. Allows preview and export

Requirements:
- Python 3.8+
- PyQt6
- moviepy
- pydub
- speech_recognition
- vosk (for offline speech recognition)
- numpy

Install dependencies:
    pip install -r requirements.txt

Note: You also need to download a Vosk language model for offline speech recognition.
Download from: https://alphacephei.com/vosk/models
For English: vosk-model-small-en-us-0.15
Extract to: models/vosk-model-small-en-us-0.15
"""

import sys
import os
import json
import wave
import numpy as np
from pathlib import Path
from datetime import timedelta

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QFileDialog, QProgressBar, QComboBox,
    QGroupBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QPixmap, QIcon, QFont, QPalette, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from pydub import AudioSegment
import speech_recognition as sr


class SubtitleWord:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class SubtitleLine:
    def __init__(self, words, start_time, end_time):
        self.words = words
        self.start_time = start_time
        self.end_time = end_time
    
    def get_text(self):
        return " ".join(w.text for w in self.words)


class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, model_path=None):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self._is_running = True
    
    def run(self):
        try:
            self.message.emit("Extracting audio...")
            self.progress.emit(10)
            audio_path = self._extract_audio()
            if not self._is_running:
                return
            self.message.emit("Converting audio...")
            self.progress.emit(20)
            wav_path = self._convert_to_wav(audio_path)
            if not self._is_running:
                return
            self.message.emit("Generating subtitles...")
            self.progress.emit(30)
            subtitle_lines = self._generate_subtitles(wav_path)
            self.progress.emit(90)
            self.message.emit("Complete!")
            self.progress.emit(100)
            self.processing_complete.emit(subtitle_lines)
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    
    def stop(self):
        self._is_running = False
    
    def _extract_audio(self):
        video = VideoFileClip(self.video_path)
        audio_path = os.path.join(
            os.path.dirname(self.video_path),
            f"{os.path.splitext(os.path.basename(self.video_path))[0]}_audio.mp3"
        )
        video.audio.write_audiofile(audio_path, codec='mp3')
        video.close()
        return audio_path
    
    def _convert_to_wav(self, audio_path):
        audio = AudioFileClip(audio_path)
        wav_path = os.path.join(
            os.path.dirname(audio_path),
            f"{os.path.splitext(os.path.basename(audio_path))[0]}.wav"
        )
        audio.write_audiofile(wav_path, codec='pcm_s16le')
        audio.close()
        return wav_path
    
    def _generate_subtitles(self, wav_path):
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        import vosk
        model = vosk.Model(self.model_path)
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Audio must be mono, 16-bit PCM WAV")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        subtitle_lines = []
        current_line_words = []
        current_line_start = 0
        while self._is_running:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "result" in result:
                    for item in result["result"]:
                        word = item.get("word", "")
                        start = item.get("start", 0)
                        end = item.get("end", 0)
                        if word:
                            current_line_words.append(SubtitleWord(word, start, end))
                            if (end - current_line_start > 5.0) or len(current_line_words) >= 10:
                                if current_line_words:
                                    line = SubtitleLine(current_line_words, current_line_start, current_line_words[-1].end_time)
                                    subtitle_lines.append(line)
                                    current_line_words = []
                                    current_line_start = end
        if current_line_words:
            line = SubtitleLine(current_line_words, current_line_start, current_line_words[-1].end_time)
            subtitle_lines.append(line)
        final_result = json.loads(rec.FinalResult())
        if "result" in final_result:
            for item in final_result["result"]:
                word = item.get("word", "")
                start = item.get("start", 0)
                end = item.get("end", 0)
                if word:
                    current_line_words.append(SubtitleWord(word, start, end))
            if current_line_words:
                line = SubtitleLine(current_line_words, current_line_start, current_line_words[-1].end_time)
                subtitle_lines.append(line)
        wf.close()
        return subtitle_lines


class VideoExporter(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    export_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path, subtitle_lines, output_path, subtitle_style=None):
        super().__init__()
        self.video_path = video_path
        self.subtitle_lines = subtitle_lines
        self.output_path = output_path
        self.subtitle_style = subtitle_style or {}
        self._is_running = True
    
    def run(self):
        try:
            self.message.emit("Loading video...")
            self.progress.emit(10)
            video = VideoFileClip(self.video_path)
            self.message.emit("Generating subtitles...")
            self.progress.emit(30)
            subtitle_clips = self._create_subtitle_clips(video)
            if not self._is_running:
                video.close()
                return
            self.message.emit("Compositing...")
            self.progress.emit(50)
            final_video = CompositeVideoClip([video] + subtitle_clips)
            self.message.emit("Exporting...")
            self.progress.emit(70)
            final_video.write_videofile(
                self.output_path,
                codec='libx264',
                audio_codec='aac',
                fps=video.fps,
                threads=4,
                preset='fast'
            )
            self.progress.emit(100)
            self.message.emit("Done!")
            self.export_complete.emit(self.output_path)
            video.close()
            final_video.close()
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    
    def stop(self):
        self._is_running = False
    
    def _create_subtitle_clips(self, video):
        subtitle_clips = []
        font = self.subtitle_style.get('font', 'Arial-Bold')
        fontsize = self.subtitle_style.get('fontsize', 40)
        color = self.subtitle_style.get('color', 'white')
        highlight_color = self.subtitle_style.get('highlight_color', 'yellow')
        stroke_color = self.subtitle_style.get('stroke_color', 'black')
        stroke_width = self.subtitle_style.get('stroke_width', 2)
        for line in self.subtitle_lines:
            position = ('center', 0.85)
            for word in line.words:
                word_clip = TextClip(
                    word.text, fontsize=fontsize, font=font, color=color,
                    stroke_color=stroke_color, stroke_width=stroke_width,
                    bg_color='transparent'
                ).set_position(position).set_start(word.start_time)
                highlighted_clip = TextClip(
                    word.text, fontsize=fontsize, font=font, color=highlight_color,
                    stroke_color=stroke_color, stroke_width=stroke_width,
                    bg_color='transparent'
                ).set_position(position).set_start(word.start_time).set_duration(word.end_time - word.start_time)
                subtitle_clips.append(word_clip)
                subtitle_clips.append(highlighted_clip)
        return subtitle_clips


class SubtitlePreviewWidget(QWidget):
    def __init__(self, subtitle_lines, parent=None):
        super().__init__(parent)
        self.subtitle_lines = subtitle_lines
        self.current_time = 0
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.subtitle_label = QLabel("")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: rgba(0,0,0,150); padding: 20px; border-radius: 10px;")
        self.layout.addWidget(self.subtitle_label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_subtitle)
    
    def start(self, interval=50):
        self.timer.start(interval)
    
    def stop(self):
        self.timer.stop()
    
    def set_time(self, time):
        self.current_time = time
        self.update_subtitle()
    
    def update_subtitle(self):
        if not self.subtitle_lines:
            self.subtitle_label.setText("")
            return
        active_line = None
        for line in self.subtitle_lines:
            if line.start_time <= self.current_time <= line.end_time:
                active_line = line
                break
        if active_line:
            html = "<div style='text-align: center;'>"
            for word in active_line.words:
                if word.start_time <= self.current_time <= word.end_time:
                    html += f"<span style='background-color: yellow; color: black; padding: 2px 4px;'>{word.text}</span> "
                else:
                    html += f"<span>{word.text}</span> "
            html += "</div>"
            self.subtitle_label.setText(html)
        else:
            self.subtitle_label.setText("")


class VideoEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Editor - Highlighted Subtitles")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #aaa; }
            QLabel { color: #ccc; }
            QPushButton { background-color: #444; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; min-width: 120px; }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { background-color: #333; color: #666; }
            QPushButton[primary="true"] { background-color: #0078d7; }
            QPushButton[primary="true"]:hover { background-color: #0095ff; }
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; background-color: #333; }
            QProgressBar::chunk { background-color: #0078d7; border-radius: 3px; }
            QComboBox { background-color: #444; color: #fff; border: 1px solid #555; padding: 5px; border-radius: 4px; }
            QScrollArea { border: 1px solid #444; background-color: #333; }
        """)
        self.video_path = None
        self.subtitle_lines = []
        self.processor = None
        self.exporter = None
        self.model_path = None
        self.init_ui()
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self.on_media_position_changed)
    
    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        header = QLabel("Video Editor with Highlighted Subtitles")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        content_split = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout()
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background-color: #000;")
        self.video_widget.setMinimumSize(640, 360)
        video_layout.addWidget(self.video_widget)
        video_controls = QHBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self.open_video)
        video_controls.addWidget(self.btn_open)
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_play.setEnabled(False)
        video_controls.addWidget(self.btn_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setEnabled(False)
        video_controls.addWidget(self.btn_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        video_controls.addWidget(self.btn_stop)
        video_layout.addLayout(video_controls)
        video_group.setLayout(video_layout)
        left_panel.addWidget(video_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        left_panel.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa;")
        left_panel.addWidget(self.status_label)
        content_split.addLayout(left_panel, 60)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        subtitles_group = QGroupBox("Subtitles")
        subtitles_layout = QVBoxLayout()
        self.subtitle_preview = SubtitlePreviewWidget([])
        self.subtitle_preview.setMinimumHeight(200)
        subtitles_layout.addWidget(self.subtitle_preview)
        self.subtitle_list = QScrollArea()
        self.subtitle_list_widget = QWidget()
        self.subtitle_list_layout = QVBoxLayout(self.subtitle_list_widget)
        self.subtitle_list_layout.setSpacing(5)
        self.subtitle_list.setWidget(self.subtitle_list_widget)
        self.subtitle_list.setWidgetResizable(True)
        self.subtitle_list.setMinimumHeight(200)
        subtitles_layout.addWidget(self.subtitle_list)
        subtitles_group.setLayout(subtitles_layout)
        right_panel.addWidget(subtitles_group)
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Vosk Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Select model...")
        self.model_combo.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Font Size:"))
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["24", "32", "40", "48", "56"])
        self.font_size_combo.setCurrentText("40")
        style_layout.addWidget(self.font_size_combo)
        settings_layout.addLayout(style_layout)
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        self.btn_generate = QPushButton("Generate Subtitles")
        self.btn_generate.setProperty("primary", "true")
        self.btn_generate.clicked.connect(self.generate_subtitles)
        self.btn_generate.setEnabled(False)
        export_layout.addWidget(self.btn_generate)
        self.btn_preview = QPushButton("Preview with Subtitles")
        self.btn_preview.clicked.connect(self.preview_with_subtitles)
        self.btn_preview.setEnabled(False)
        export_layout.addWidget(self.btn_preview)
        self.btn_export = QPushButton("Export Video")
        self.btn_export.setProperty("primary", "true")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        export_group.setLayout(export_layout)
        right_panel.addWidget(export_group)
        content_split.addLayout(right_panel, 40)
        main_layout.addLayout(content_split)
        self.scan_vosk_models()
    
    def scan_vosk_models(self):
        model_dirs = ["models", os.path.join(os.path.expanduser("~"), "vosk-models"), os.path.join(os.path.dirname(__file__), "models")]
        found_models = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path) and "vosk" in item.lower():
                        found_models.append(item_path)
        self.model_combo.clear()
        if found_models:
            for model in found_models:
                self.model_combo.addItem(model)
            self.model_combo.setCurrentIndex(0)
            self.model_path = found_models[0]
        else:
            self.model_combo.addItem("No models found")
            self.status_label.setText("Warning: No Vosk models found. Download from https://alphacephei.com/vosk/models")
    
    def on_model_selected(self, index):
        if index > 0:
            self.model_path = self.model_combo.currentText()
            self.status_label.setText(f"Model selected: {os.path.basename(self.model_path)}")
    
    def open_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            self.load_video(file_dialog.selectedFiles()[0])
    
    def load_video(self, video_path):
        self.video_path = video_path
        self.stop_video()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_generate.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.subtitle_lines = []
        self.update_subtitle_list()
        self.subtitle_preview.stop()
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.status_label.setText(f"Loaded: {os.path.basename(video_path)}")
    
    def play_video(self):
        if self.video_path:
            self.media_player.play()
            self.btn_play.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            if self.subtitle_lines:
                self.subtitle_preview.start()
    
    def pause_video(self):
        self.media_player.pause()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.subtitle_preview.stop()
    
    def stop_video(self):
        self.media_player.stop()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.subtitle_preview.stop()
        self.subtitle_preview.set_time(0)
    
    def on_media_position_changed(self, position):
        if self.video_path:
            duration = self.media_player.duration()
            if duration > 0:
                self.progress_bar.setValue(int((position / duration) * 100))
            if self.subtitle_lines:
                self.subtitle_preview.set_time(position / 1000)
    
    def generate_subtitles(self):
        if not self.video_path:
            self.status_label.setText("Error: No video loaded")
            return
        if not self.model_path or not os.path.exists(self.model_path):
            self.status_label.setText("Error: No valid Vosk model")
            return
        self.btn_generate.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.processor = VideoProcessor(self.video_path, self.model_path)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.message.connect(self.status_label.setText)
        self.processor.processing_complete.connect(self.on_subtitles_generated)
        self.processor.error_occurred.connect(self.on_processing_error)
        self.processor.start()
    
    def on_subtitles_generated(self, subtitle_lines):
        self.subtitle_lines = subtitle_lines
        self.update_subtitle_list()
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.status_label.setText(f"Subtitles: {len(subtitle_lines)} lines")
    
    def on_processing_error(self, error):
        self.status_label.setText(error)
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_play.setEnabled(True if self.video_path else False)
    
    def update_subtitle_list(self):
        for i in reversed(range(self.subtitle_list_layout.count())):
            widget = self.subtitle_list_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        for i, line in enumerate(self.subtitle_lines):
            label = QLabel(f"{i+1}. [{self._format_time(line.start_time)} - {self._format_time(line.end_time)}] {line.get_text()}")
            label.setStyleSheet("padding: 5px; border-bottom: 1px solid #444;")
            label.setWordWrap(True)
            self.subtitle_list_layout.addWidget(label)
    
    def _format_time(self, seconds):
        return str(timedelta(seconds=int(seconds)))
    
    def preview_with_subtitles(self):
        if not self.video_path or not self.subtitle_lines:
            return
        self.stop_video()
        import tempfile
        temp_dir = tempfile.mkdtemp()
        preview_path = os.path.join(temp_dir, "preview.mp4")
        fontsize = int(self.font_size_combo.currentText())
        self.exporter = VideoExporter(
            self.video_path, self.subtitle_lines, preview_path,
            subtitle_style={'fontsize': fontsize, 'color': 'white', 'highlight_color': 'yellow', 'stroke_color': 'black', 'stroke_width': 2}
        )
        self.exporter.progress.connect(self.progress_bar.setValue)
        self.exporter.message.connect(self.status_label.setText)
        self.exporter.export_complete.connect(lambda p: self._on_preview_ready(p, temp_dir))
        self.exporter.error_occurred.connect(self.on_processing_error)
        self.btn_generate.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.exporter.start()
    
    def _on_preview_ready(self, video_path, temp_dir):
        self.load_video(video_path)
        self.media_player.play()
        if self.exporter:
            self.exporter.deleteLater()
            self.exporter = None
        self._temp_dir = temp_dir
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.status_label.setText("Preview ready")
    
    def export_video(self):
        if not self.video_path or not self.subtitle_lines:
            return
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("MP4 Video (*.mp4)")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setDefaultSuffix("mp4")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            output_path = file_dialog.selectedFiles()[0]
            fontsize = int(self.font_size_combo.currentText())
            self.exporter = VideoExporter(
                self.video_path, self.subtitle_lines, output_path,
                subtitle_style={'fontsize': fontsize, 'color': 'white', 'highlight_color': 'yellow', 'stroke_color': 'black', 'stroke_width': 2}
            )
            self.exporter.progress.connect(self.progress_bar.setValue)
            self.exporter.message.connect(self.status_label.setText)
            self.exporter.export_complete.connect(self.on_export_complete)
            self.exporter.error_occurred.connect(self.on_processing_error)
            self.btn_generate.setEnabled(False)
            self.btn_open.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.exporter.start()
    
    def on_export_complete(self, output_path):
        if self.exporter:
            self.exporter.deleteLater()
            self.exporter = None
        self.btn_generate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.btn_play.setEnabled(True if self.video_path else False)
        self.btn_preview.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.status_label.setText(f"Exported: {os.path.basename(output_path)}")
    
    def closeEvent(self, event):
        if self.processor and self.processor.isRunning():
            self.processor.stop()
            self.processor.wait()
        if self.exporter and self.exporter.isRunning():
            self.exporter.stop()
            self.exporter.wait()
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoEditorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
