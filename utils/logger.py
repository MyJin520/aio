import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class LoggerManager:
    """统一的日志管理器"""

    _instances = {}

    @classmethod
    def get_logger(cls, service_name: str = "default", log_level: str = "INFO") -> Dict[str, logging.Logger]:
        """获取日志器实例"""
        if service_name not in cls._instances:
            cls._instances[service_name] = cls._setup_logger(service_name, log_level)
        return cls._instances[service_name]

    @staticmethod
    def _setup_logger(service_name: str, log_level: str) -> Dict[str, logging.Logger]:
        """配置日志系统"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(os.getcwd(), "logs", today_str)
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)

        # 创建不同角色的日志器
        loggers = {}

        # 主日志器
        main_logger = logging.getLogger(f"{service_name}_main")
        main_logger.setLevel(level)
        main_logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        main_logger.addHandler(console_handler)

        # 文件处理器
        log_file = os.path.join(log_dir, f"{service_name}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        main_logger.addHandler(file_handler)

        loggers['main'] = main_logger

        # 访问日志器
        access_logger = logging.getLogger(f"{service_name}_access")
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False

        access_file = os.path.join(log_dir, f"{service_name}_access.log")
        access_handler = RotatingFileHandler(
            access_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        access_handler.setFormatter(file_formatter)
        access_logger.addHandler(access_handler)

        loggers['access'] = access_logger

        # 错误日志器
        error_logger = logging.getLogger(f"{service_name}_error")
        error_logger.setLevel(logging.ERROR)
        error_logger.propagate = False

        error_file = os.path.join(log_dir, f"{service_name}_error.log")
        error_handler = RotatingFileHandler(
            error_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_logger.addHandler(error_handler)

        loggers['error'] = error_logger

        # 抑制第三方库日志
        for lib in ["funasr", "fish_speech", "torch", "werkzeug", "modelscope", "urllib3"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

        # 配置werkzeug日志
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.WARNING)

        return loggers


def get_logger(service_name: str = "default", logger_type: str = "main") -> logging.Logger:
    """获取指定类型的日志器"""
    loggers = LoggerManager.get_logger(service_name)
    return loggers.get(logger_type, loggers['main'])