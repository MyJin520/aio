import os
import tempfile
import time

import numpy as np
import torch
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from pydub import AudioSegment

from config.tts_config import Config


class TTSServer:
    def __init__(self, config: Config, loggers):
        self.config = config
        self.tts_engine = None
        self.is_initialized = False
        self.initialization_error = None
        self.logger = loggers['main_logger']
        self.access_logger = loggers['access_logger']
        self.error_logger = loggers['error_logger']

    def prepare_model(self):
        # 检查文件是否存在
        files_exist = self.config.llama_ckpt_file.exists() and self.config.decoder_ckpt_path.exists()

        if not files_exist:
            error_msg = "模型文件不完整，请检查是否包含 models.pth 和 codec.pth"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info("模型文件验证通过")
        self.logger.info(f"LLaMA 模型路径: {self.config.llama_ckpt_file}")
        self.logger.info(f"解码器模型路径: {self.config.decoder_ckpt_path}")
        return True

    def init_engine(self):
        """初始化TTS引擎"""
        if self.is_initialized:  # 避免重复初始化
            return
        try:
            # 设备配置优化
            device_obj = torch.device(self.config.device)
            dtype = torch.float16 if self.config.device == "cuda" else torch.float32

            # 仅在CUDA可用时进行性能优化
            if device_obj.type == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()

            self.logger.info(f"加载 LLaMA 模型...")
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.config.model_dir,
                device=device_obj,
                precision=dtype,
                compile=self.config.compile_model,
            )

            self.logger.info(f"加载解码器模型...")
            decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=self.config.decoder_ckpt_path,
                device=device_obj,
            )

            self.logger.info(f"初始化 TTS 推理引擎...")
            self.tts_engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                compile=self.config.compile_model,
                precision=dtype,
            )

            self.is_initialized = True
            self.logger.info("TTS引擎初始化成功")
            self.logger.info(f"设备: {self.config.device} | 编译优化: {self.config.compile_model}")
        except Exception as e:
            self.initialization_error = str(e)
            self.error_logger.error(f"引擎初始化失败: {self.initialization_error}")
            self.is_initialized = False

    def generate_tts(self, text: str, refs: list = None, request_id: str = None):
        """生成TTS音频，优化音频处理流程（输出MP3格式）"""
        if not self.is_initialized:
            raise RuntimeError(f"引擎未初始化: {self.initialization_error}")

        references = []
        if refs:
            references = [
                ServeReferenceAudio(audio=ref["audio_data"], text=ref.get("text", ""))
                for ref in refs
                if ref.get("audio_data")
            ]

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
        self.logger.info(f"开始合成文本: {text[:50]}...")
        start_time = time.time()
        audio_segments = []

        try:
            for result in self.tts_engine.inference(req):
                if result.code == "error":
                    raise Exception(result.error)
                if result.audio and result.audio[1] is not None:
                    audio_segments.append(result.audio[1])
        except Exception as e:
            self.error_logger.error(f"TTS推理失败 (request_id: {request_id}): {str(e)}")
            raise

        if not audio_segments:
            raise Exception("未生成音频数据")

        # 音频处理
        audio_data = np.concatenate(audio_segments, axis=0, dtype=np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-6:
            audio_data /= max_val  # 原地归一化，减少内存占用

        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file_path = temp_file.name

        audio_segment = AudioSegment(
            data=audio_data_int16.tobytes(),
            sample_width=audio_data_int16.dtype.itemsize,
            frame_rate=self.tts_engine.decoder_model.sample_rate,
            channels=1
        )

        audio_segment.export(temp_file_path, format="mp3", bitrate="128k")

        processing_time = time.time() - start_time
        audio_size_kb = os.path.getsize(temp_file_path) / 1024
        self.logger.info(
            f"合成完成 | 耗时: {processing_time:.2f}s | 音频大小: {audio_size_kb:.1f}KB | 格式: MP3")

        return temp_file_path, processing_time, audio_size_kb

    def get_status(self):
        """获取服务状态"""
        status = "ready" if self.is_initialized else "not_ready"
        resp = {
            "status": status,
            "compile_enabled": self.config.compile_model,
            "model_dir": str(self.config.model_dir),
            "device": self.config.device,
            "log_dir": str(self.config.log_dir)
        }
        if not self.is_initialized:
            resp["error"] = self.initialization_error
        return resp, 200 if self.is_initialized else 503

    def start_compile(self):
        """用于服务启动时开始编译模型"""
        if not self.is_initialized:
            raise RuntimeError(f"引擎未初始化: {self.initialization_error}")

        req = ServeTTSRequest(
            text="你好世界",
            max_new_tokens=2048,
            top_p=0.7,
            temperature=0.7,
            repetition_penalty=1.0,
            streaming=False
        )
        start_time = time.time()
        audio_segments = []
        try:
            for result in self.tts_engine.inference(req):
                if result.code == "error":
                    raise Exception(result.error)
                if result.audio and result.audio[1] is not None:
                    audio_segments.append(result.audio[1])
        except Exception as e:
            self.error_logger.error(f"初始化编译TTS推理失败: {str(e)}")
            raise

        if not audio_segments:
            raise Exception("未生成音频数据")

        audio_data = np.concatenate(audio_segments, axis=0, dtype=np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-6:
            audio_data /= max_val

        processing_time = time.time() - start_time

        self.logger.info(f"初始化编译完成 | 耗时: {processing_time:.2f}s | 音频处理后的元素个数: {audio_data.size}")
