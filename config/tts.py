import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from config import BaseConfig


@dataclass
class TTSConfig(BaseConfig):
    """TTS服务配置"""
    # 服务标识
    service_name: str = "tts"

    # 模型配置
    model_path: Optional[Path] = None  # 直接使用 Path 类型注解
    device: str = "cuda"
    compile_model: bool = True
    decoder_ckpt_path: Optional[Path] = None
    llama_ckpt_file: Optional[Path] = None

    # 音频配置
    output_format: str = "mp3"
    bitrate: str = "192k"

    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__()

        # 设置默认模型目录（如果未提供）
        if self.model_path is None:
            self.model_path = (Path(__file__).parent.parent / "tts_model").resolve()

        # 确保模型目录存在
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 设置模型文件路径
        self.decoder_ckpt_path = self.model_path / "codec.pth"
        self.llama_ckpt_file = self.model_path / "model.pth"
