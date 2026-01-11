import os
import sys
import collections
import threading
import queue
import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from funasr import AutoModel
from pydub import AudioSegment

from config.asr import ASRConfig
from services.base import BaseService
from utils.audio import AudioUtils
from utils.sse import SSEHelper


class ASRService(BaseService):
    """ASR语音识别服务"""

    def __init__(self, config: ASRConfig, logger):
        super().__init__(config, logger)

        # 禁用第三方库冗余输出
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["FUNASR_VERBOSE"] = "0"

        # 状态变量初始化
        self.chunk_size_samples = int(config.sample_rate * config.chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.sse_queue = queue.Queue(maxsize=config.sse_queue_maxsize)

        # 识别状态
        self.recording_active = False
        self.listen_mode = False
        self.text_buffer = collections.deque(maxlen=3)
        self.current_text = ""
        self.final_results: List[str] = []
        self.listen_results: List[str] = []

        # 静音检测状态
        self.last_voice_time = time.time()
        self.waiting_for_silence = False
        self.silence_timeout_ended = False

        # 模型与音频流
        self.model: Optional[AutoModel] = None
        self.model_cache: Dict[str, Any] = {}
        self.audio_stream: Optional[Any] = None

        # 音频收集
        self.audio_fragments = []
        self.recognition_thread: Optional[threading.Thread] = None

        # 打印配置信息
        self._log_config()

    def _log_config(self) -> None:
        """打印初始化配置"""
        self.logger.info("=" * 60)
        self.logger.info("📋 ASR 服务初始化配置")
        self.logger.info(f"触发关键词: 开始='{self.config.start_keyword}', 结束='{self.config.stop_keyword}'")
        self.logger.info(f"静音检测: 阈值={self.config.silence_threshold}, 超时={self.config.silence_timeout_seconds}s")
        self.logger.info(f"音频配置: {self.config.sample_rate}Hz / {self.config.chunk_duration_ms}ms/块")
        self.logger.info("=" * 60)

    def _audio_callback(self, indata: np.ndarray, frames, time, status) -> None:
        """音频采集回调"""
        if status:
            self.logger.warning(f"⚠️ 音频状态异常: {status}")
        try:
            audio_data = indata[:, 0].copy().astype(np.float32)
            self.audio_queue.put(audio_data)

            if self.recording_active:
                audio_segment = AudioUtils.convert_numpy_to_audio_segment(audio_data)
                self.audio_fragments.append(audio_segment)

        except Exception as e:
            self.logger.error(f"❌ 音频回调错误: {str(e)}")

    def _validate_model_path(self, model_path: str) -> bool:
        """验证模型路径有效性"""
        if not model_path:
            self.logger.error("❌ 模型路径未配置")
            return False

        if not os.path.exists(model_path):
            self.logger.error(f"❌ 模型路径不存在: {model_path}")
            return False

        if not os.path.isdir(model_path):
            self.logger.error(f"❌ 模型路径不是目录: {model_path}")
            return False

        missing = [f for f in self.config.required_model_files
                   if not os.path.exists(os.path.join(model_path, f))]
        if missing:
            self.logger.error(f"❌ 模型路径缺失必要文件: {missing}")
            return False

        self.logger.info(f"✅ 模型路径验证成功: {model_path}")
        return True

    def load_model(self) -> None:
        """加载ASR模型"""
        if self.config.model_path and self._validate_model_path(self.config.model_path):
            load_strategies = [
                {"model": self.config.model_path, "hub": "local"},
                {"model": self.config.model_path}
            ]

            for idx, strategy in enumerate(load_strategies):
                try:
                    self.logger.info(f"🔍 尝试加载模型 (策略{idx + 1}): {strategy}")
                    self.model = AutoModel(
                        **strategy,
                        disable_pbar=True,
                        disable_update=True
                    )
                    self.logger.info("✅ 模型加载成功")
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️ 策略{idx + 1}加载失败: {str(e)[:100]}")

            raise RuntimeError(f"所有本地模型加载策略均失败: {self.config.model_path}")
        else:
            raise RuntimeError("未配置有效的模型路径")

    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        """处理音频块识别"""
        if self.model is None:
            return ""

        try:
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    res = self.model.generate(
                        input=audio_chunk,
                        cache=self.model_cache,
                        is_final=is_final,
                        chunk_size=self.config.chunk_size,
                        encoder_chunk_look_back=self.config.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.config.decoder_chunk_look_back
                    )
                finally:
                    sys.stdout = old_stdout

            if not res or 'text' not in res[0]:
                return ""

            return res[0]['text'].strip()
        except Exception as e:
            self.logger.error(f"❌ 音频识别错误: {str(e)}")
            return ""

    def _recognition_worker(self) -> None:
        """识别工作线程"""
        if self.model is None:
            self.load_model()

        audio_buffer = np.array([], dtype=np.float32)
        self.logger.info("🎤 实时语音识别线程已启动")

        try:
            while not self.stop_event.is_set():
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)

                    # 静音超时处理
                    if self.recording_active:
                        self._handle_silence_timeout(audio_chunk)
                        if self.silence_timeout_ended:
                            self._reset_recognition_state()
                            audio_buffer = np.array([], dtype=np.float32)
                            continue

                    # 音频缓冲区处理
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                    while len(audio_buffer) >= self.chunk_size_samples:
                        process_chunk = audio_buffer[:self.chunk_size_samples]
                        audio_buffer = audio_buffer[self.chunk_size_samples:]

                        # 音频识别
                        recognized_text = self._process_audio_chunk(process_chunk)
                        if not recognized_text:
                            continue

                        # 关键词检测
                        self.text_buffer.append(recognized_text)
                        start_detected, stop_detected = self._check_keywords()

                        # 状态控制
                        self._handle_recognition_state(start_detected, stop_detected, recognized_text)

                        # 实时结果推送
                        if self.recording_active and recognized_text != self.current_text:
                            self.current_text = recognized_text
                            self._log_realtime_text(recognized_text)
                            SSEHelper.send_sse_data(self.sse_queue, 'partial', recognized_text)

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"❌ 识别线程异常: {str(e)}")
                    continue

        finally:
            self._process_remaining_audio(audio_buffer)
            SSEHelper.clear_sse_queue(self.sse_queue, self.logger)

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """检测是否为静音"""
        return AudioUtils.is_silent(audio_chunk, self.config.silence_threshold)

    def _check_keywords(self) -> Tuple[bool, bool]:
        """检查文本缓冲区中是否包含开始或结束关键词"""
        combined_text = "".join(self.text_buffer)
        start_detected = self.config.start_keyword in combined_text
        stop_detected = self.config.stop_keyword in combined_text
        return start_detected, stop_detected

    def _reset_recognition_state(self) -> None:
        """重置识别状态"""
        self.recording_active = False
        self.listen_mode = False
        self.waiting_for_silence = False
        self.silence_timeout_ended = False
        self.current_text = ""
        self.text_buffer.clear()

        # 发送结束事件
        if self.final_results:
            SSEHelper.send_sse_data(self.sse_queue, 'final',
                                    " ".join(self.final_results))
            self.final_results.clear()

        # 清空音频片段
        if self.audio_fragments:
            self.audio_fragments.clear()

    def _handle_silence_timeout(self, audio_chunk: np.ndarray) -> None:
        """处理静音超时"""
        if not self._is_silent(audio_chunk):
            self.last_voice_time = time.time()
            self.waiting_for_silence = False
        else:
            silence_duration = time.time() - self.last_voice_time
            if silence_duration > self.config.silence_timeout_seconds:
                if not self.waiting_for_silence:
                    self.waiting_for_silence = True
                    self.logger.info("🕐 检测到静音超时，等待结束...")
                else:
                    self.silence_timeout_ended = True
                    self.logger.info("⏹️ 静音超时，自动结束识别")

    def _handle_recognition_state(self, start_detected: bool, stop_detected: bool,
                                  recognized_text: str) -> None:
        """处理识别状态变更"""
        if start_detected and not self.recording_active:
            self.recording_active = True
            self.logger.info(f"▶️ 检测到开始关键词: '{self.config.start_keyword}'，开始录音")
            SSEHelper.send_sse_data(self.sse_queue, 'status', 'recording_started')

        elif stop_detected and self.recording_active:
            self.recording_active = False
            self.logger.info(f"⏹️ 检测到结束关键词: '{self.config.stop_keyword}'，停止录音")

            # 处理最终结果
            if recognized_text:
                self.final_results.append(recognized_text)
                if self.listen_mode:
                    self.listen_results.append(recognized_text)

            # 发送最终结果
            if self.final_results:
                final_text = " ".join(self.final_results)
                SSEHelper.send_sse_data(self.sse_queue, 'final', final_text)

                # 保存音频
                if self.audio_fragments:
                    AudioUtils.merge_audio_segments(
                        self.audio_fragments,
                        self.config.audio_output_path,
                        logger=self.logger
                    )
                    self.audio_fragments.clear()

            # 重置状态
            self._reset_recognition_state()

    def _process_remaining_audio(self, audio_buffer: np.ndarray) -> None:
        """处理剩余的音频缓冲区"""
        if len(audio_buffer) > 0 and self.recording_active:
            # 处理剩余的音频
            final_text = self._process_audio_chunk(audio_buffer, is_final=True)
            if final_text:
                self.final_results.append(final_text)
                final_result = " ".join(self.final_results)
                SSEHelper.send_sse_data(self.sse_queue, 'final', final_result)

                # 保存音频
                if self.audio_fragments:
                    AudioUtils.merge_audio_segments(
                        self.audio_fragments,
                        self.config.audio_output_path,
                        logger=self.logger
                    )
                    self.audio_fragments.clear()

    def _log_realtime_text(self, text: str) -> None:
        """记录实时识别文本"""
        self.logger.info(f"🎤 实时识别: {text}")

    def start(self) -> None:
        """启动ASR服务"""
        try:
            with self.thread_lock:
                if self.is_running:
                    self.logger.warning("服务已在运行中")
                    return

                # 初始化音频设备
                AudioUtils.list_audio_devices(self.logger)

                # 启动识别线程
                self.recognition_thread = threading.Thread(
                    target=self._recognition_worker,
                    daemon=True,
                    name="ASR_Worker"
                )
                self.recognition_thread.start()

                # 启动音频流
                self.audio_stream = AudioUtils.start_audio_stream(
                    self.config.sample_rate,
                    self.config.chunk_duration_ms,
                    self._audio_callback,
                    self.logger
                )

                self.is_running = True
                self.logger.info("✅ ASR服务启动成功")

        except Exception as e:
            self.logger.error(f"❌ ASR服务启动失败: {str(e)}")
            raise

    def stop(self) -> None:
        """停止ASR服务"""
        with self.thread_lock:
            if not self.is_running:
                return

            self.stop_event.set()

            # 关闭音频流
            if self.audio_stream:
                try:
                    if hasattr(self.audio_stream, 'active') and self.audio_stream.active:
                        self.audio_stream.stop()
                    self.audio_stream.close()
                except Exception:
                    pass

            # 释放模型
            if self.model:
                try:
                    del self.model
                except Exception:
                    pass

            # 等待线程结束
            if self.recognition_thread:
                self.recognition_thread.join(timeout=3.0)

            # 合并音频片段
            if self.audio_fragments:
                AudioUtils.merge_audio_segments(
                    self.audio_fragments,
                    self.config.audio_output_path,
                    logger=self.logger
                )

            SSEHelper.clear_sse_queue(self.sse_queue, self.logger)
            self.is_running = False
            self.logger.info("✅ ASR服务已停止")

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "status": "running" if self.is_running else "stopped",
            "service": "asr",
            "model_loaded": self.model is not None,
            "recording_active": self.recording_active,
            "listen_mode": self.listen_mode,
            "sse_queue_size": self.sse_queue.qsize()
        }