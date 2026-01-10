import logging
import os
import sys
from datetime import datetime
from logging import Filter, Handler, LogRecord
from typing import Dict, Optional


class LevelFilter(Filter):
    """日志级别过滤器"""
    def __init__(self, level: int, exact: bool = True):
        super().__init__()
        self.level = level
        self.exact = exact

    def filter(self, record: LogRecord) -> bool:
        if self.exact:
            return record.levelno == self.level
        return record.levelno >= self.level


def setup_logger() -> logging.Logger:
    """初始化全局日志器，按级别分离日志文件"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join(os.getcwd(), "logs", today_str)
    os.makedirs(log_dir, exist_ok=True)

    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # 格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建不同级别的日志文件
    info_log_file = os.path.join(log_dir, "asr_server.log")
    warning_log_file = os.path.join(log_dir, "warning.log")
    error_log_file = os.path.join(log_dir, "error.log")

    # 为每个级别创建文件处理器
    handlers: Dict[str, Handler] = {}

    # INFO 级别处理器 - 包含INFO、WARNING、ERROR、CRITICAL
    info_handler = logging.FileHandler(info_log_file, encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    info_handler.addFilter(LevelFilter(logging.INFO, exact=False))
    handlers['info'] = info_handler

    # WARNING 级别处理器 - 只包含WARNING
    warning_handler = logging.FileHandler(warning_log_file, encoding='utf-8')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    warning_handler.addFilter(LevelFilter(logging.WARNING, exact=True))
    handlers['warning'] = warning_handler

    # ERROR 级别处理器 - 包含ERROR和CRITICAL
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(LevelFilter(logging.ERROR, exact=False))
    handlers['error'] = error_handler

    # 控制台处理器（显示所有级别）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # 添加所有处理器到根日志器
    for handler in handlers.values():
        root_logger.addHandler(handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("asr_server")
    logger.info(f"日志目录: {log_dir}")
    logger.info("日志文件: asr_server.log (INFO+), warning.log (WARNING), error.log (ERROR+)")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取指定名称的日志器"""
    if name:
        return logging.getLogger(f"asr_server.{name}")
    return logging.getLogger("asr_server")