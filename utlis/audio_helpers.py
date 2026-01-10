import numpy as np
import sounddevice as sd

from config.asr_config import ASRConfig


def is_silent(audio_chunk: np.ndarray, silence_threshold: float) -> bool:
    """检测是否为静音"""
    energy = np.sqrt(np.mean(audio_chunk ** 2))
    return energy < silence_threshold


def list_audio_devices(logger) -> None:
    """列出并验证音频设备"""
    logger.info("🔊 可用音频输入设备:")
    devices = sd.query_devices()
    input_devices = [(i, dev) for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]

    if not input_devices:
        raise RuntimeError("未找到可用的音频输入设备")

    # 打印设备列表
    for i, dev in input_devices:
        logger.info(f"   {i}: {dev['name']} (通道数: {dev['max_input_channels']})")

    # 验证默认设备
    default_id = sd.default.device[0]
    if default_id not in [i for i, _ in input_devices]:
        logger.warning(f"默认设备{default_id}不可用，切换到第一个可用设备")
        sd.default.device = input_devices[0][0]

    logger.info(f"使用音频设备: {sd.query_devices(sd.default.device[0])['name']}")


def start_audio_stream(config: ASRConfig, audio_queue, audio_callback, logger):
    """启动音频流"""
    try:
        audio_stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=int(config.sample_rate * config.chunk_duration_ms / 1000),
            device=sd.default.device[0]
        )
        audio_stream.start()
        logger.info("音频流已启动")
        return audio_stream
    except Exception as e:
        logger.error(f"音频流启动失败: {str(e)}")
        raise
