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
    model_dir: str = None  # 这里应该是 Path 类型，或者需要在 __post_init__ 中转换
    device: str = "cuda"
    model_id: str = "fishaudio/openaudio-s1-mini"
    compile_model: bool = True
    decoder_ckpt_path: Optional[Path] = None
    llama_ckpt_file: Optional[Path] = None

    # 音频配置
    output_format: str = "mp3"
    bitrate: str = "192k"

    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__()

        # 将字符串转换为 Path 对象
        from pathlib import Path
        import os

        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

        # 设置默认模型目录（如果未提供）
        if self.model_dir is None:
            current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
            self.model_dir = (current_dir / "models").resolve()

        # 确保模型目录存在
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 设置模型文件路径
        self.decoder_ckpt_path = self.model_dir / "codec.pth"
        self.llama_ckpt_file = self.model_dir / "model.pth"