from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASRConfig:
    """ASR配置封装"""
    # 音频配置
    sample_rate: int = 16000
    chunk_duration_ms: int = 600
    # 模型配置
    model_name: str = "paraformer-zh-streaming"
    model_revision: str = "v2.0.4"
    model_path: Optional[str] = None
    model_id: str = 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online'
    required_model_files: List[str] = field(default_factory=lambda: ['config.yaml', 'model.pt', 'tokens.json'])
    # 识别配置
    start_keyword: str = "开始"
    stop_keyword: str = "结束"
    chunk_size: List[int] = field(default_factory=lambda: [0, 10, 5])
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    # 静音检测
    silence_threshold: float = 0.001
    silence_timeout_seconds: float = 7.0
    # SSE配置
    sse_queue_maxsize: int = 100
    min_output_interval: float = 0.1