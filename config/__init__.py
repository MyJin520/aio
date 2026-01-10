from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path
import os


@dataclass
class BaseConfig:
    """基础配置类"""
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "INFO"

    # CORS配置
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self):
        """后初始化处理"""
        pass