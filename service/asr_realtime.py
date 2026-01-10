import collections

import logging
import os
import queue
import sys
import threading
import time

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from funasr import AutoModel

# 新增音频处理依赖
from pydub import AudioSegment

from config.asr_config import ASRConfig
from utlis.audio_helpers import list_audio_devices, start_audio_stream
from utlis.logger import get_logger
from utlis.sse_helpers import clear_sse_queue, send_sse_data


class RealTimeASR:
    def __init__(self, config: ASRConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.audio_logger = get_logger("audio")
        self.model_logger = get_logger("model")
        self.recognition_logger = get_logger("recognition")

        # 禁用第三方库冗余输出
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["FUNASR_VERBOSE"] = "0"

        # 状态变量初始化
        self.chunk_size_samples = int(config.sample_rate * config.chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()  # 全局停止信号
        self.sse_queue = queue.Queue(maxsize=config.sse_queue_maxsize)

        # 识别状态
        self.recording_active = False  # 是否正在识别
        self.listen_mode = False  # 是否为Listen模式
        self.text_buffer = collections.deque(maxlen=3)  # 关键词检测缓冲区
        self.current_text = ""  # 当前识别文本
        self.final_results: List[str] = []  # 普通模式结果
        self.listen_results: List[str] = []  # Listen模式结果

        # 静音检测状态
        self.last_voice_time = time.time()
        self.waiting_for_silence = False
        self.silence_timeout_ended = False

        # 模型与缓存
        self.model: Optional[AutoModel] = None
        self.model_cache: Dict[str, Any] = {}
        self.audio_stream: Optional[Any] = None

        # 收集音频片段
        self.audio_fragments = []  # 收集识别过程中的音频片段
        self.audio_sample_rate = config.sample_rate  # 采样率
        self.audio_channels = 1  # 单声道
        self.audio_sample_width = 2  # 16位音频（2字节/采样点）
        self.audio_output_path = "tmp.mp3"  # 输出MP3文件路径

        # 打印配置信息
        self._log_config()

    def _log_config(self) -> None:
        """打印初始化配置"""
        self.logger.info("=" * 60)
        self.logger.info("📋 ASR 服务初始化配置")
        self.logger.info(f"模型路径: {self.config.model_path} (仅使用本地模型，不自动下载)")
        self.logger.info(f"触发关键词: 开始='{self.config.start_keyword}', 结束='{self.config.stop_keyword}'")
        self.logger.info(f"静音检测: 阈值={self.config.silence_threshold}, 超时={self.config.silence_timeout_seconds}s")
        self.logger.info(f"音频配置: {self.config.sample_rate}Hz / {self.config.chunk_duration_ms}ms/块")
        self.logger.info(f"音频保存: 识别停止后将保存为 {self.audio_output_path}")
        self.logger.info("=" * 60)

    def _convert_numpy_audio_to_segment(self, audio_data: np.ndarray) -> AudioSegment:
        try:
            # 将float32（-1~1）转换为int16（-32768~32767）
            audio_int16 = (audio_data * 32767).astype(np.int16)
            # 转换为字节流
            audio_bytes = audio_int16.tobytes()
            # 创建AudioSegment对象
            audio_segment = AudioSegment(
                data=audio_bytes,
                sample_width=self.audio_sample_width,
                frame_rate=self.audio_sample_rate,
                channels=self.audio_channels
            )
            return audio_segment
        except Exception as e:
            self.audio_logger.error(f"❌ 音频格式转换失败: {str(e)}")
            raise

    def _merge_audio_fragments(self) -> None:
        """
        合并收集的音频片段为MP3文件
        """
        if not self.audio_fragments:
            self.audio_logger.warning("⚠️ 无音频片段可合并")
            return

        try:
            # 合并所有音频片段
            merged_audio = AudioSegment.empty()
            for fragment in self.audio_fragments:
                merged_audio += fragment

            original_dBFS = merged_audio.dBFS
            original_max_dBFS = merged_audio.max_dBFS
            self.audio_logger.info(f"📊 原始音频统计: dBFS={original_dBFS:.1f}, max_dBFS={original_max_dBFS:.1f}")

            if original_dBFS < -40:  # 如果音量极低
                volume_gain_db = 25  # 大幅提升25dB
                self.audio_logger.info(f"📈 检测到极低音量，应用大幅增益: +{volume_gain_db}dB")
                merged_audio = merged_audio + volume_gain_db

                # 标准化到目标音量
                target_dBFS = -16.0  # 目标音量级别
                current_dBFS = merged_audio.dBFS
                if current_dBFS < target_dBFS:
                    needed_gain = target_dBFS - current_dBFS
                    if needed_gain > 0:
                        self.audio_logger.info(f"🎯 进一步标准化增益: +{needed_gain:.1f}dB (目标: {target_dBFS}dBFS)")
                        merged_audio = merged_audio + min(needed_gain, 15)  # 限制最大增益15dB避免削波
            else:
                volume_gain_db = 12  # 常规提升12dB
                self.audio_logger.info(f"📈 常规音量增益: +{volume_gain_db}dB")
                merged_audio = merged_audio + volume_gain_db
                merged_audio = merged_audio.normalize(headroom=1.0)  # 标准化，保留1dB headroom

            # 防止削波保护
            max_possible = merged_audio.max
            if max_possible >= 32767:  # 16位音频最大值
                self.audio_logger.warning(f"⚠️ 检测到削波风险! 当前最大值: {max_possible}")
                # 降低音量直到不削波
                while merged_audio.max >= 32767 and volume_gain_db > 0:
                    volume_gain_db -= 2
                    merged_audio = merged_audio - 2
                    self.audio_logger.warning(f"⚠️ 降低音量以避免削波，新增益: {volume_gain_db}dB")

            # 最终音量检查和微调
            final_dBFS = merged_audio.dBFS
            final_max_dBFS = merged_audio.max_dBFS

            self.audio_logger.info(f"📊 处理后音频统计: dBFS={final_dBFS:.1f}, max_dBFS={final_max_dBFS:.1f}")

            # 如果仍然太低，再次尝试提升
            if final_dBFS < -30:
                additional_gain = min(10, -30 - final_dBFS)  # 最多再提升10dB
                if additional_gain > 0:
                    self.audio_logger.info(f"📈 二次增益: +{additional_gain:.1f}dB")
                    merged_audio = merged_audio + additional_gain
                    final_dBFS = merged_audio.dBFS
                    final_max_dBFS = merged_audio.max_dBFS
                    self.audio_logger.info(f"📊 二次处理后: dBFS={final_dBFS:.1f}, max_dBFS={final_max_dBFS:.1f}")
            merged_audio.export(
                self.audio_output_path,
                format="mp3",
                bitrate="192k",  # 提高比特率
                parameters=["-q:a", "0"]  # 最高质量
            )

            final_duration = len(merged_audio) / 1000
            self.audio_logger.info(f"✅ 音频已保存为: {self.audio_output_path} (时长: {final_duration:.2f}s)")
            self.audio_logger.info(f"🎯 最终音量: {final_dBFS:.1f}dBFS (目标范围: -25 ~ -10 dBFS)")
            self.audio_logger.info(f"🔊 峰值音量: {final_max_dBFS:.1f}dBFS (应 < 0dBFS 避免削波)")
            if final_dBFS < -25:
                self.audio_logger.warning("⚠️ 音量仍然偏低，建议检查音频采集设备增益设置")
            if final_max_dBFS >= -1.0:
                self.audio_logger.warning("⚠️ 峰值音量接近0dBFS，可能存在轻微削波")

        except Exception as e:
            self.audio_logger.error(f"❌ 音频合并失败: {str(e)}")
            import traceback
            self.audio_logger.error(traceback.format_exc())
        finally:
            self.audio_fragments.clear()
            self.audio_logger.info("🔄 音频片段缓存已清空")

    # 模型管理
    def _validate_model_path(self, model_path: str) -> bool:
        """验证模型路径有效性"""
        if not model_path:
            self.model_logger.error("❌ 模型路径未配置（model_path为空）")
            return False

        if not os.path.exists(model_path):
            self.model_logger.error(f"❌ 模型路径不存在: {model_path}")
            return False

        # 检查是否为目录
        if not os.path.isdir(model_path):
            self.model_logger.error(f"❌ 模型路径不是目录: {model_path}")
            return False

        missing = [f for f in self.config.required_model_files if not os.path.exists(os.path.join(model_path, f))]
        if missing:
            self.model_logger.error(f"❌ 模型路径缺失必要文件: {missing}")
            return False

        self.model_logger.info(f"✅ 模型路径验证成功: {model_path}")
        return True

    def load_model(self) -> None:
        """加载模型（仅使用本地模型，不下载）"""
        # 验证模型路径
        if not self._validate_model_path(self.config.model_path):
            error_msg = f"❌ 模型路径验证失败: {self.config.model_path}，无法加载模型"
            self.model_logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 模型加载重试逻辑（仅本地加载策略）
        load_strategies = [
            {"model": self.config.model_path, "hub": "local"},
            {"model": self.config.model_path}
        ]

        for idx, strategy in enumerate(load_strategies):
            try:
                self.model_logger.info(f"🔍 尝试加载模型 (策略{idx + 1}): {strategy}")
                self.model = AutoModel(
                    **strategy,
                    disable_pbar=True,
                    disable_update=True
                )
                self.model_logger.info("✅ 模型加载成功")
                return
            except Exception as e:
                self.model_logger.warning(f"⚠️ 策略{idx + 1}加载失败: {str(e)[:100]}")

        error_msg = f"❌ 所有本地模型加载策略均失败，请检查模型路径或模型文件完整性: {self.config.model_path}"
        self.model_logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 音频处理
    def _audio_callback(self, indata: np.ndarray, frames, time, status) -> None:
        """音频采集回调"""
        if status:
            self.audio_logger.warning(f"⚠️ 音频状态异常: {status}")
        try:
            # 提取单声道音频数据
            audio_data = indata[:, 0].copy().astype(np.float32)
            self.audio_queue.put(audio_data)
            if self.recording_active:
                audio_segment = self._convert_numpy_audio_to_segment(audio_data)
                self.audio_fragments.append(audio_segment)

        except Exception as e:
            self.audio_logger.error(f"❌ 音频回调错误: {str(e)}")

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """检测是否为静音"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        if energy < self.config.silence_threshold:
            self.audio_logger.debug(f"🔇 检测到静音块，能量: {energy:.6f}")
            return True
        return False

    # 识别
    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        """处理单块音频识别"""
        if self.model is None:
            return ""

        try:
            # 重定向stdout避免第三方库输出
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    self.recognition_logger.debug(f"🎤 处理音频块，长度: {len(audio_chunk)}，is_final: {is_final}")
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

            if not res or len(res) == 0 or 'text' not in res[0]:
                self.recognition_logger.debug("📭 音频识别返回空结果")
                return ""

            text = res[0]['text'].strip()
            if text:
                self.recognition_logger.debug(f"🎯 识别结果: '{text}'")
            return text
        except Exception as e:
            self.recognition_logger.error(f"❌ 音频识别错误: {str(e)}")
            return ""

    def _check_keywords(self) -> Tuple[bool, bool]:
        """检查缓冲区中的关键词"""
        buffer_text = "".join(self.text_buffer)
        start_detected = self.config.start_keyword in buffer_text
        stop_detected = self.config.stop_keyword in buffer_text

        if start_detected:
            self.recognition_logger.info(f"🔑 检测到开始关键词: '{self.config.start_keyword}'，缓冲区: '{buffer_text}'")
        if stop_detected:
            self.recognition_logger.info(f"🔑 检测到结束关键词: '{self.config.stop_keyword}'，缓冲区: '{buffer_text}'")

        return start_detected, stop_detected

    def _reset_recognition_state(self) -> None:
        """重置识别状态"""
        self.model_cache.clear()
        self.text_buffer.clear()
        self.current_text = ""
        clear_sse_queue(self.sse_queue, self.logger)
        self.last_voice_time = time.time()
        self.waiting_for_silence = False
        self.silence_timeout_ended = False
        # 重置时清空音频片段（避免残留）
        self.audio_fragments.clear()
        self.recognition_logger.info("🔄 识别状态已重置")

    # ------------------------ 识别线程 ------------------------
    def _recognition_worker(self) -> None:
        """识别工作线程"""
        if self.model is None:
            self.load_model()

        audio_buffer = np.array([], dtype=np.float32)
        self.recognition_logger.info("🎤 实时语音识别线程已启动")

        try:
            while not self.stop_event.is_set():
                try:
                    # 获取音频块（超时避免死等）
                    audio_chunk = self.audio_queue.get(timeout=0.5)

                    # 1. 静音超时处理
                    if self.recording_active:
                        self._handle_silence_timeout(audio_chunk)
                        if self.silence_timeout_ended:
                            self._reset_recognition_state()
                            audio_buffer = np.array([], dtype=np.float32)
                            continue

                    # 2. 音频缓冲区处理
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                    while len(audio_buffer) >= self.chunk_size_samples:
                        # 提取待处理音频块
                        process_chunk = audio_buffer[:self.chunk_size_samples]
                        audio_buffer = audio_buffer[self.chunk_size_samples:]

                        # 3. 音频识别
                        recognized_text = self._process_audio_chunk(process_chunk)
                        if not recognized_text:
                            continue

                        # 4. 关键词检测
                        self.text_buffer.append(recognized_text)
                        start_detected, stop_detected = self._check_keywords()

                        # 5. 状态控制（开始/停止/Listen模式）
                        self._handle_recognition_state(start_detected, stop_detected, recognized_text)

                        # 6. 实时结果推送
                        if self.recording_active and recognized_text != self.current_text:
                            self.current_text = recognized_text
                            self._log_realtime_text(recognized_text)
                            send_sse_data(self.sse_queue, 'partial', recognized_text)

                except queue.Empty:
                    continue
                except Exception as e:
                    self.recognition_logger.error(f"❌ 识别线程异常: {str(e)}")
                    continue

        finally:
            # 处理剩余音频
            self._process_remaining_audio(audio_buffer)
            clear_sse_queue(self.sse_queue, self.logger)

    def _handle_silence_timeout(self, audio_chunk: np.ndarray) -> None:
        """处理静音超时逻辑"""
        is_silent = self._is_silent(audio_chunk)

        if is_silent:
            if not self.waiting_for_silence:
                self.waiting_for_silence = True
                self.recognition_logger.info(f"⏳ 检测到静音，{self.config.silence_timeout_seconds}秒后自动结束")
        else:
            self.last_voice_time = time.time()
            if self.waiting_for_silence:
                self.waiting_for_silence = False
                self.recognition_logger.info("🎤 检测到语音，继续识别")

        # 静音超时触发
        if self.waiting_for_silence:
            silence_duration = time.time() - self.last_voice_time
            if silence_duration > self.config.silence_timeout_seconds:
                self.recognition_logger.info(f"🔴 静音超时({silence_duration:.1f}s)，停止识别")
                self.recording_active = False
                self.silence_timeout_ended = True

                # 保存结果并推送
                if self.current_text:
                    target_results = self.listen_results if self.listen_mode else self.final_results
                    target_results.append(self.current_text)
                    send_sse_data(self.sse_queue, 'final', self.current_text)

                # 静音超时停止时合并音频
                self._merge_audio_fragments()

                # 统一发送ListenBreak，增加mode字段区分模式
                mode = "listen" if self.listen_mode else "normal"
                send_sse_data(
                    self.sse_queue,
                    'ListenBreak',
                    f'{mode.capitalize()}模式因静音超时结束',
                    mode=mode,  # 区分普通/listen模式
                    reason='silence_timeout'  # 标注结束原因
                )

                self.recognition_logger.info(f"🔄 {'Listen' if self.listen_mode else '普通'}模式回到等待状态")

    def _handle_recognition_state(self, start_detected: bool, stop_detected: bool, text: str) -> None:
        """处理识别状态切换（核心优化：普通模式结束也发送ListenBreak）"""
        # Listen模式启动
        if self.listen_mode and not self.recording_active:
            self.recognition_logger.info("🟢 Listen模式开始识别")
            self.recording_active = True
            self._reset_recognition_state()
            return

        # 普通模式开始
        if start_detected and not self.recording_active and not self.listen_mode:
            self.recognition_logger.info(f"🟢 检测到开始关键词'{self.config.start_keyword}'，开始识别")
            self.recording_active = True
            self._reset_recognition_state()
            return

        # 停止识别
        if stop_detected and self.recording_active:
            self.recognition_logger.info(f"🔴 检测到结束关键词'{self.config.stop_keyword}'，停止识别")
            self.recording_active = False
            self.silence_timeout_ended = False

            # 保存结果
            if self.current_text:
                target_results = self.listen_results if self.listen_mode else self.final_results
                target_results.append(self.current_text)
                send_sse_data(self.sse_queue, 'final', self.current_text)

            # 关键词停止时，合并音频
            self._merge_audio_fragments()

            # 重置状态
            self._reset_recognition_state()

            # 统一发送ListenBreak，区分模式和原因
            if self.listen_mode:
                self.listen_mode = False
                send_sse_data(
                    self.sse_queue,
                    'ListenBreak',
                    'Listen模式因检测到结束关键词结束',
                    mode='listen',
                    reason='stop_keyword'
                )
                self.recognition_logger.info("🔄 Listen模式回到等待状态（关键词触发）")
            else:
                send_sse_data(
                    self.sse_queue,
                    'ListenBreak',
                    f'普通模式因检测到结束关键词结束',
                    mode='normal',
                    reason='stop_keyword'
                )
                self.recognition_logger.info(f"🔄 普通模式等待下一个开始关键词'{self.config.start_keyword}'")

    def _process_remaining_audio(self, audio_buffer: np.ndarray) -> None:
        """处理剩余音频"""
        if len(audio_buffer) == 0:
            return

        self.recognition_logger.info("⏳ 处理剩余音频...")
        # 补零到最小块大小
        if len(audio_buffer) < self.chunk_size_samples:
            padding_length = self.chunk_size_samples - len(audio_buffer)
            audio_buffer = np.pad(audio_buffer, (0, padding_length))
            self.recognition_logger.debug(f"📝 音频补零 {padding_length} 个采样点")

        # 最终识别
        text = self._process_audio_chunk(audio_buffer[:self.chunk_size_samples], is_final=True)
        if text:
            clean_text = text.replace(self.config.start_keyword, '').replace(self.config.stop_keyword, '').strip()
            target_results = self.listen_results if self.listen_mode else self.final_results
            target_results.append(text)

            self._log_realtime_text(f"剩余音频识别: {clean_text}")
            send_sse_data(self.sse_queue, 'partial', clean_text)

            mode = "Listen" if self.listen_mode else "普通"
            self.recognition_logger.info(f"🔄 {mode}模式回到等待状态")

    def _log_realtime_text(self, text: str) -> None:
        """打印实时识别文本"""
        clean_text = text.replace(self.config.start_keyword, '').replace(self.config.stop_keyword, '').strip()
        if clean_text:
            self.recognition_logger.info(f"[{time.strftime('%H:%M:%S')}] 🔄 实时识别: {clean_text}")

    def start(self) -> None:
        """启动ASR服务"""
        try:
            # 初始化音频设备
            list_audio_devices(self.logger)

            # 启动识别线程
            self.recognition_thread = threading.Thread(
                target=self._recognition_worker,
                daemon=True,
                name="ASR_Worker"
            )
            self.recognition_thread.start()

            # 启动音频流
            self.audio_stream = start_audio_stream(
                self.config,
                self.audio_queue,
                self._audio_callback,
                self.logger
            )

            # 等待停止信号
            while not self.stop_event.is_set():
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("🛑 检测到中断信号，正在停止服务...")
            # 程序中断时合并音频
            self._merge_audio_fragments()
        except Exception as e:
            self.logger.error(f"❌ ASR服务启动失败: {str(e)}")
            raise
        finally:
            self.cleanup()

    def start_listen_mode(self) -> None:
        """启动Listen模式"""
        if self.listen_mode:
            raise RuntimeError("已在Listen模式中，请等待当前会话结束")

        self.listen_mode = True
        self.listen_results.clear()
        self.recording_active = False
        self._reset_recognition_state()

        self.logger.info(
            f"🟢 Listen模式已启动，说出'{self.config.stop_keyword}'或静音{self.config.silence_timeout_seconds}秒结束")

    def get_listen_results(self) -> List[str]:
        """获取Listen模式清理后的结果"""
        return [
            r.replace(self.config.start_keyword, '').replace(self.config.stop_keyword, '').strip()
            for r in self.listen_results if r.strip()
        ]

    def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("🧹 开始清理资源...")

        # 停止事件
        self.stop_event.set()

        # 关闭音频流
        if self.audio_stream:
            try:
                if hasattr(self.audio_stream, 'active') and self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self.logger.info("✅ 音频流已关闭")
            except Exception as e:
                self.logger.warning(f"⚠️ 音频流关闭失败: {str(e)}")

        # 释放模型
        if self.model:
            try:
                del self.model
                self.logger.info("✅ 模型资源已释放")
            except Exception as e:
                self.logger.warning(f"⚠️ 模型释放失败: {str(e)}")

        # 清空队列
        clear_sse_queue(self.sse_queue, self.logger)

        # 等待线程结束
        if hasattr(self, 'recognition_thread'):
            self.recognition_thread.join(timeout=3.0)

        self.logger.info("✅ 资源清理完成")
