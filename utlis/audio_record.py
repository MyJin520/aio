import os
import threading
import wave

import numpy as np


class AudioRecorder:
    """音频录制器"""

    def __init__(self, sample_rate=16000, channels=1, dtype=np.float32, gain_factor=2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.gain_factor = gain_factor  # 音频增益系数
        self.is_recording = False
        self.audio_data = []
        self.recording_lock = threading.Lock()

    def start_recording(self):
        """开始录制音频"""
        with self.recording_lock:
            self.is_recording = True
            self.audio_data = []
            print(f"开始录音，采样率: {self.sample_rate}Hz, 增益: {self.gain_factor}x")

    def stop_recording(self):
        """停止录制音频"""
        with self.recording_lock:
            self.is_recording = False
            print(f"停止录音，共录制 {len(self.audio_data)} 个音频块")

    def add_audio_chunk(self, audio_chunk):
        """添加音频块到录制数据"""
        with self.recording_lock:
            if self.is_recording:
                amplified_chunk = audio_chunk * self.gain_factor
                # 防止音频削波（clipping）
                amplified_chunk = np.clip(amplified_chunk, -1.0, 1.0)
                self.audio_data.append(amplified_chunk.copy())

    def save_recording(self, filepath):
        """保存录制的音频到文件"""
        with self.recording_lock:
            if not self.audio_data:
                print("没有音频数据可保存")
                return False

            # 合并所有音频块
            combined_audio = np.concatenate(self.audio_data)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 将浮点数据转换为16位整数
            audio_int16 = (combined_audio * 32767).astype(np.int16)

            # 写入WAV文件
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            print(f"录音已保存到: {filepath}")
            return True
