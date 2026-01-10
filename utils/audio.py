import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from typing import Optional, Callable
import os


class AudioUtils:
    """音频处理工具类"""

    @staticmethod
    def list_audio_devices(logger) -> None:
        """列出并验证音频设备"""
        logger.info("🔊 可用音频输入设备:")
        devices = sd.query_devices()
        input_devices = [(i, dev) for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]

        if not input_devices:
            raise RuntimeError("未找到可用的音频输入设备")

        for i, dev in input_devices:
            logger.info(f"   {i}: {dev['name']} (通道数: {dev['max_input_channels']})")

        default_id = sd.default.device[0]
        if default_id not in [i for i, _ in input_devices]:
            logger.warning(f"默认设备{default_id}不可用，切换到第一个可用设备")
            sd.default.device = input_devices[0][0]

        logger.info(f"使用音频设备: {sd.query_devices(sd.default.device[0])['name']}")

    @staticmethod
    def start_audio_stream(sample_rate: int, chunk_duration_ms: int, callback: Callable, logger):
        """启动音频流"""
        try:
            audio_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=callback,
                blocksize=int(sample_rate * chunk_duration_ms / 1000),
                device=sd.default.device[0]
            )
            audio_stream.start()
            logger.info("音频流已启动")
            return audio_stream
        except Exception as e:
            logger.error(f"音频流启动失败: {str(e)}")
            raise

    @staticmethod
    def is_silent(audio_chunk: np.ndarray, silence_threshold: float) -> bool:
        """检测是否为静音"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy < silence_threshold

    @staticmethod
    def convert_numpy_to_audio_segment(
            audio_data: np.ndarray,
            sample_rate: int = 16000,
            channels: int = 1,
            sample_width: int = 2
    ) -> AudioSegment:
        """将numpy数组转换为AudioSegment"""
        try:
            # 将float32（-1~1）转换为int16（-32768~32767）
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            audio_segment = AudioSegment(
                data=audio_bytes,
                sample_width=sample_width,
                frame_rate=sample_rate,
                channels=channels
            )
            return audio_segment
        except Exception as e:
            raise ValueError(f"音频格式转换失败: {str(e)}")

    @staticmethod
    def merge_audio_segments(
            segments: list,
            output_path: str,
            target_dBFS: float = -16.0,
            logger=None
    ) -> bool:
        """合并音频片段并优化音量"""
        if not segments:
            if logger:
                logger.warning("⚠️ 无音频片段可合并")
            return False

        try:
            # 合并所有音频片段
            merged_audio = AudioSegment.empty()
            for segment in segments:
                merged_audio += segment

            # 音量优化
            original_dBFS = merged_audio.dBFS
            if original_dBFS < -40:
                volume_gain_db = 25
                if logger:
                    logger.info(f"📈 检测到极低音量，应用大幅增益: +{volume_gain_db}dB")
                merged_audio = merged_audio + volume_gain_db

            # 标准化到目标音量
            current_dBFS = merged_audio.dBFS
            if current_dBFS < target_dBFS:
                needed_gain = target_dBFS - current_dBFS
                if needed_gain > 0:
                    if logger:
                        logger.info(f"🎯 应用标准化增益: +{needed_gain:.1f}dB")
                    merged_audio = merged_audio + min(needed_gain, 15)

            # 防止削波
            max_possible = merged_audio.max
            if max_possible >= 32767:
                if logger:
                    logger.warning(f"⚠️ 检测到削波风险! 当前最大值: {max_possible}")
                while merged_audio.max >= 32767:
                    merged_audio = merged_audio - 2

            # 导出音频
            merged_audio.export(
                output_path,
                format=os.path.splitext(output_path)[1][1:],  # 从扩展名获取格式
                bitrate="192k",
                parameters=["-q:a", "0"]
            )

            if logger:
                final_duration = len(merged_audio) / 1000
                final_dBFS = merged_audio.dBFS
                logger.info(f"✅ 音频已保存: {output_path} (时长: {final_duration:.2f}s)")
                logger.info(f"🎯 最终音量: {final_dBFS:.1f}dBFS")

            return True

        except Exception as e:
            if logger:
                logger.error(f"❌ 音频合并失败: {str(e)}")
            return False