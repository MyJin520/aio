from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from config import BaseConfig


@dataclass
class ASRConfig(BaseConfig):
    """ASR服务配置"""
    # 服务标识
    service_name: str = "asr"

    # 音频配置
    sample_rate: int = 16000
    chunk_duration_ms: int = 600
    audio_channels: int = 1

    # 模型配置
    model_path: Optional[Path] = None
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

    # 音频输出
    audio_output_path: str = "tmp.mp3"

    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__()

        # 设置默认模型目录（如果未提供）
        if self.model_path is None:
            self.model_path = (Path(__file__).parent.parent / "asr_model").resolve()
        else:
            self.model_path = Path(self.model_path).resolve()

        # 确保模型目录存在
        self.model_path.mkdir(parents=True, exist_ok=True)
