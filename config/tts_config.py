import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from datetime import datetime


class Config:
    model_dir: Path = None
    device: str = "cuda"
    host: str = "localhost"
    port: int = 5000
    model_id: str = "fishaudio/openaudio-s1-mini"
    compile_model: bool = True
    decoder_ckpt_path: Path = None
    llama_ckpt_file: Path = None
    log_dir: Path = None

    @classmethod
    def from_args(cls, args):
        config = cls()

        # 获取当前程序所在目录
        current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent

        # 设置 model_dir 默认值为当前程序路径的 models 目录
        if args.model_dir:
            config.model_dir = Path(args.model_dir).resolve()
        else:
            config.model_dir = (current_dir / "models").resolve()

        # 设置日志目录为 logs/年-月-日/
        today = datetime.now().strftime("%Y-%m-%d")
        config.log_dir = (current_dir / "logs" / today).resolve()

        config.device = args.device
        config.host = args.host
        config.port = args.port
        config.model_id = args.model_id
        config.compile_model = args.compile

        # 确保模型目录存在
        config.model_dir.mkdir(parents=True, exist_ok=True)
        # 确保日志目录存在
        config.log_dir.mkdir(parents=True, exist_ok=True)

        config.decoder_ckpt_path = config.model_dir / "codec.pth"
        config.llama_ckpt_file = config.model_dir / "model.pth"

        return config


def setup_logging(config: Config):
    """配置日志系统，按照日期目录存放日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 格式器
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    access_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    error_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(pathname)s:%(lineno)d | %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 主日志文件处理器 (按大小轮转)
    main_log_file = config.log_dir / "tts_server.log"
    main_file_handler = RotatingFileHandler(
        str(main_log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    main_file_handler.setFormatter(formatter)
    logger.addHandler(main_file_handler)

    # 访问日志处理器 (记录接口调用)
    access_log_file = config.log_dir / "access.log"
    access_logger = logging.getLogger('access')
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False  # 不传递给根日志器

    access_file_handler = RotatingFileHandler(
        str(access_log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    access_file_handler.setFormatter(access_formatter)
    access_logger.addHandler(access_file_handler)

    # 错误日志处理器
    error_log_file = config.log_dir / "error.log"
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False

    error_file_handler = RotatingFileHandler(
        str(error_log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    error_file_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_file_handler)

    # 抑制冗余日志
    for lib in ["fish_speech", "torch", "werkzeug", "modelscope", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # 配置werkzeug日志 (Flask内置)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)  # 关闭werkzeug的详细日志

    return {
        'main_logger': logger,
        'access_logger': access_logger,
        'error_logger': error_logger
    }


def parse_arguments():
    # 获取当前程序所在目录
    current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

    parser = argparse.ArgumentParser(description="Fish-Speech TTS Server")
    parser.add_argument("--model_dir", type=str, help="Model directory path (default: current_dir/models)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model_id", type=str, default="fishaudio/openaudio-s1-mini")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    return args