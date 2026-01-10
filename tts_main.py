
from pathlib import Path
import sys

from api.tts_routers import APIServer
from config.tts_config import parse_arguments, Config, setup_logging
from service.tts_engine import TTSServer


def main():
    # 解析参数
    args = parse_arguments()
    config = Config.from_args(args)

    # 设置日志
    loggers = setup_logging(config)
    logger = loggers['main_logger']
    error_logger = loggers['error_logger']

    # 显示配置信息
    logger.info("=== 启动配置信息 ===")
    logger.info(f"程序目录: {Path(__file__).parent.parent.resolve()}")
    logger.info(f"模型目录: {config.model_dir}")
    logger.info(f"日志目录: {config.log_dir}")
    logger.info(f"设备: {config.device}")
    logger.info(f"主机: {config.host}:{config.port}")
    logger.info(f"模型ID: {config.model_id}")
    logger.info(f"编译优化: {config.compile_model}")
    logger.info(f"LLaMA模型路径: {config.llama_ckpt_file}")
    logger.info(f"解码器模型路径: {config.decoder_ckpt_path}")
    logger.info("=" * 60)

    # 检查模型目录是否存在
    if not config.model_dir.exists():
        logger.info(f"模型目录不存在，将自动创建: {config.model_dir}")
        config.model_dir.mkdir(parents=True, exist_ok=True)

    # 初始化TTS服务器
    tts_server = TTSServer(config, loggers)

    # 准备模型并初始化引擎
    try:
        tts_server.prepare_model()
        tts_server.init_engine()
    except Exception as e:
        error_logger.error(f"初始化失败: {str(e)}")
        logger.error(f"程序启动失败，退出...")
        sys.exit(1)

    # 检查是否初始化成功
    if not tts_server.is_initialized:
        logger.error(f"引擎初始化失败: {tts_server.initialization_error}")
        sys.exit(1)

    # 初始化模型加速
    tts_server.start_compile()

    # 创建并启动API服务器
    api_server = APIServer(tts_server)
    logger.info(f"✅ 服务启动成功 | http://{config.host}:{config.port}")
    logger.info(f"📊 服务状态: http://{config.host}:{config.port}/tts/status")
    logger.info(f"🔧 日志文件位置:")
    logger.info(f"   📁 {config.log_dir}/tts_server.log    # 主日志")
    logger.info(f"   📁 {config.log_dir}/access.log       # 接口访问日志")
    logger.info(f"   📁 {config.log_dir}/error.log        # 错误日志")
    logger.info(f"🚀 准备接收请求...")

    try:
        api_server.run(host=config.host, port=config.port)
    except KeyboardInterrupt:
        logger.info("🛑 服务被手动停止")
    except Exception as e:
        error_logger.error(f"服务运行异常: {str(e)}")
        logger.error("服务异常退出")
        sys.exit(1)


if __name__ == "__main__":
    main()