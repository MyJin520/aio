import threading
import queue
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseService(ABC):
    """基础服务类"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.stop_event = threading.Event()
        self.thread_lock = threading.Lock()
        self.is_running = False

    @abstractmethod
    def start(self) -> None:
        """启动服务"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止服务"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        pass

    def cleanup(self) -> None:
        """清理资源"""
        with self.thread_lock:
            if self.is_running:
                self.stop()
                self.is_running = False