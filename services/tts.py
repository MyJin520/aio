import os
import threading
import time
from io import BytesIO
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from pydub import AudioSegment

from config.tts import TTSConfig
from services.base import BaseService


class TTSService(BaseService):
    """TTS文本转语音服务"""

    def __init__(self, config: TTSConfig, logger):
        super().__init__(config, logger)

        self.tts_engine = None
        self.engine_lock = threading.Lock()
        self.initialization_error = None

    def initialize(self) -> None:
        """初始化TTS引擎"""
        with self.thread_lock:
            if self.is_running:
                return

            try:
                # 检查模型文件
                files_exist = (self.config.llama_ckpt_file.exists() and
                               self.config.decoder_ckpt_path.exists())

                if not files_exist:
                    error_msg = "模型文件不完整，请检查是否包含 model.pth 和 codec.pth"
                    raise FileNotFoundError(error_msg)

                self.logger.info("✅ 模型文件验证通过")

                # 设备配置
                device_obj = torch.device(self.config.device)
                dtype = torch.float16 if self.config.device == "cuda" else torch.float32

                if device_obj.type == "cuda":
                    torch.backends.cudnn.benchmark = True
                    torch.cuda.empty_cache()

                # 加载模型
                self.logger.info("🔍 加载LLaMA模型...")
                llama_queue = launch_thread_safe_queue(
                    checkpoint_path=self.config.model_dir,
                    device=device_obj,
                    precision=dtype,
                    compile=self.config.compile_model,
                )

                self.logger.info("🔍 加载解码器模型...")
                decoder_model = load_decoder_model(
                    config_name="modded_dac_vq",
                    checkpoint_path=self.config.decoder_ckpt_path,
                    device=device_obj,
                )

                self.logger.info("🔍 初始化TTS推理引擎...")
                self.tts_engine = TTSInferenceEngine(
                    llama_queue=llama_queue,
                    decoder_model=decoder_model,
                    compile=self.config.compile_model,
                    precision=dtype,
                )

                self.is_running = True
                self.logger.info("✅ TTS引擎初始化成功")

            except Exception as e:
                self.initialization_error = str(e)
                self.logger.error(f"❌ TTS引擎初始化失败: {self.initialization_error}")
                raise

    def _inference(self, text: str, references: List[ServeReferenceAudio] = None, request_id: Optional[str] = None) -> tuple:
        """核心推理逻辑（私有方法）"""
        if not self.is_running:
            raise RuntimeError("TTS服务未初始化")

        references = references or []

        # 构建请求
        req = ServeTTSRequest(
            text=text,
            references=references,
            max_new_tokens=2048,
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.0,
            streaming=False
        )

        # 推理生成音频
        self.logger.info(f"🎙️ 开始合成文本: {text[:50]}...")
        start_time = time.time()
        audio_segments = []

        try:
            with self.engine_lock:
                for result in self.tts_engine.inference(req):
                    if result.code == "error":
                        raise Exception(result.error)
                    if result.audio and result.audio[1] is not None:
                        audio_segments.append(result.audio[1])
        except Exception as e:
            self.logger.error(f"❌ TTS推理失败: {str(e)}")
            raise

        if not audio_segments:
            raise Exception("未生成音频数据")

        # 音频处理
        audio_data = np.concatenate(audio_segments, axis=0, dtype=np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-6:
            audio_data /= max_val

        processing_time = time.time() - start_time
        return audio_data, processing_time

    def generate_speech(self, text: str, refs: list = None, request_id: str = None) -> tuple:
        """生成语音"""
        # 处理参考音频
        references = []
        if refs:
            references = [
                ServeReferenceAudio(audio=ref["audio_data"], text=ref.get("text", ""))
                for ref in refs if ref.get("audio_data")
            ]

        # 调用核心推理方法
        audio_data, processing_time = self._inference(text, references, request_id)

        # 音频数据转换为int16
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # 使用内存流替代临时文件
        audio_stream = BytesIO()

        audio_segment = AudioSegment(
            data=audio_data_int16.tobytes(),
            sample_width=audio_data_int16.dtype.itemsize,
            frame_rate=self.tts_engine.decoder_model.sample_rate,
            channels=1
        )

        audio_segment.export(
            audio_stream,
            format="mp3",
            bitrate=self.config.bitrate
        )

        # 重置流指针到开始位置
        audio_stream.seek(0)

        audio_size_kb = len(audio_stream.getvalue()) / 1024

        self.logger.info(f"✅ 合成完成 | 耗时: {processing_time:.2f}s | 大小: {audio_size_kb:.1f}KB")

        return audio_stream, processing_time, audio_size_kb

    def start(self) -> None:
        """启动TTS服务"""
        self.initialize()

    def stop(self) -> None:
        """停止TTS服务"""
        with self.thread_lock:
            if not self.is_running:
                return

            # 释放资源
            if self.tts_engine:
                try:
                    del self.tts_engine
                except Exception:
                    pass

            self.is_running = False
            self.logger.info("✅ TTS服务已停止")

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "status": "ready" if self.is_running else "not_ready",
            "service": "tts",
            "compile_enabled": self.config.compile_model,
            "model_dir": str(self.config.model_dir),
            "device": self.config.device
        }

    def init_engine_compile(self):
        """初始化引擎编译（预热）"""
        # 调用核心推理方法，使用固定文本
        audio_data, processing_time = self._inference("你好世界")

        audio_size_kb = len(audio_data) / 1024
        self.logger.info(f"✅ 首次编译 | 耗时: {processing_time:.2f}s | 大小: {audio_size_kb:.1f}KB")
