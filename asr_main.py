import logging
import threading
import sys
import atexit
import os

from api.asr_routes import register_routes
from config.asr_config import ASRConfig
from service.asr_realtime import RealTimeASR
from utlis.cli import parse_args
from utlis.logger import setup_logger


def create_flask_app(asr_instance, logger, port):
    """创建Flask应用"""
    from flask import Flask

    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    # 注册路由
    register_routes(app, asr_instance, logger)

    # 打印启动信息
    def log_startup():
        logger.info("\n ASR服务已启动")
        logger.info(f" 访问地址: http://0.0.0.0:{port}")
        logger.info("可用接口:")
        logger.info("   GET  /asr/status      - 服务状态")
        logger.info("   GET  /asr/config      - 配置信息")
        logger.info("   GET  /asr/results     - 识别结果")
        logger.info("   GET  /asr/stream      - 实时SSE流")
        logger.info("   POST /asr/listen      - 启动Listen模式")
        logger.info("   POST /asr/clear-sse   - 清空SSE队列")
        logger.info("=" * 60)

    return app, log_startup


def main():
    """主入口"""
    # 解析参数
    args = parse_args()

    # 获取当前程序所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_models_dir = os.path.join(current_dir, 'models')

    # 如果没有指定model_path，使用默认的models目录
    if args.model_path is None:
        args.model_path = default_models_dir
        os.makedirs(default_models_dir, exist_ok=True)
        print(f"使用默认模型目录: {default_models_dir}")

    # 初始化日志
    logger = setup_logger()

    # 初始化ASR配置
    asr_config = ASRConfig(
        start_keyword=args.start_keyword,
        stop_keyword=args.stop_keyword,
        silence_threshold=args.silence_threshold,
        silence_timeout_seconds=args.silence_timeout,
        model_path=args.model_path,
        model_name=args.model_name,
        model_revision=args.model_revision
    )

    # 初始化ASR核心
    asr = RealTimeASR(asr_config, logger)

    try:
        # 加载模型
        asr.load_model()

        # 创建Flask应用
        flask_app, log_startup = create_flask_app(asr, logger, args.port)

        # 注册退出清理
        atexit.register(lambda: (asr.stop_event.set(), asr.cleanup()))

        # 启动ASR（后台线程）
        asr_thread = threading.Thread(target=asr.start, daemon=True, name="ASR_Main")
        asr_thread.start()

        # 启动Flask
        log_startup()
        flask_app.run(
            host='0.0.0.0',
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        asr.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
