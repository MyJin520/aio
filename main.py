#!/usr/bin/env python3
"""
AI语音服务主程序
整合ASR（语音识别）和TTS（文本转语音）功能
"""

import signal
import sys

from flask import Flask
from waitress import serve
from api.router import VoiceServiceRouter
from config.asr import ASRConfig
from config.tts import TTSConfig
from services.asr import ASRService
from services.tts import TTSService
from utils.logger import LoggerManager
from utils.cli import parse_args


class VoiceService:
    """语音服务管理器"""

    def __init__(self, args):
        self.args = args
        self.asr_service = None
        self.tts_service = None
        self.app = None
        self.loggers = None

        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """信号处理函数"""
        self.loggers['main'].info(f"接收到信号 {signum}，正在停止服务...")
        self.stop()
        sys.exit(0)

    def initialize_services(self):
        """初始化服务"""
        # 初始化日志
        self.loggers = LoggerManager.get_logger("voice_service", self.args.log_level)
        main_logger = self.loggers['main']

        main_logger.info("=" * 60)
        main_logger.info("🚀 AI语音服务启动")
        main_logger.info("=" * 60)

        # 初始化ASR服务
        if self.args.enable_asr:
            try:
                main_logger.info("🔍 初始化ASR服务...")
                asr_config = ASRConfig(
                    host=self.args.host,
                    port=self.args.port,
                    start_keyword=self.args.start_keyword,
                    stop_keyword=self.args.stop_keyword,
                    silence_threshold=self.args.silence_threshold,
                    silence_timeout_seconds=self.args.silence_timeout,
                    log_level=self.args.log_level
                )

                self.asr_service = ASRService(asr_config, main_logger)
                self.asr_service.start()
                main_logger.info("✅ ASR服务初始化成功")

            except Exception as e:
                main_logger.error(f"❌ ASR服务初始化失败: {e}")
                if not self.args.ignore_errors:
                    raise

        # 初始化TTS服务
        if self.args.enable_tts:
            try:
                main_logger.info("🔍 初始化TTS服务...")
                tts_config = TTSConfig(
                    host=self.args.host,
                    port=self.args.tts_port or self.args.port,
                    model_path=self.args.tts_model_path,
                    device=self.args.device,
                    compile_model=self.args.compile,
                    log_level=self.args.log_level
                )

                self.tts_service = TTSService(tts_config, main_logger)
                self.tts_service.start()
                main_logger.info("✅ TTS服务初始化成功")
                # 初始化引擎编译
                self.tts_service.init_engine_compile()
            except Exception as e:
                main_logger.error(f"❌ TTS服务初始化失败: {e}")
                if not self.args.ignore_errors:
                    raise

    def create_flask_app(self):
        """创建Flask应用"""
        main_logger = self.loggers['main']

        self.app = Flask(__name__)

        # 注册路由
        router = VoiceServiceRouter(
            self.app,
            self.asr_service,
            self.tts_service,
            main_logger
        )

        return self.app

    def print_startup_info(self):
        """打印启动信息"""
        main_logger = self.loggers['main']

        main_logger.info("\n📡 服务信息:")
        main_logger.info(f"   访问地址: http://{self.args.host}:{self.args.port}")
        main_logger.info(f"   ASR服务: {'启用' if self.args.enable_asr else '禁用'}")
        main_logger.info(f"   TTS服务: {'启用' if self.args.enable_tts else '禁用'}")

        if self.args.enable_asr:
            main_logger.info("\n🎤 ASR接口:")
            main_logger.info("   GET  /asr/status      - ASR服务状态")
            main_logger.info("   POST /asr/listen      - 启动Listen模式")
            main_logger.info("   GET  /asr/stream      - 实时SSE流")
            main_logger.info("   GET  /asr/audio       - 获取录音文件")

        if self.args.enable_tts:
            main_logger.info("\n🎙️ TTS接口:")
            main_logger.info("   POST /tts/create      - 生成语音")
            main_logger.info("   GET  /tts/status      - TTS服务状态")

        main_logger.info("\n🔧 通用接口:")
        main_logger.info("   GET  /health         - 服务健康检查")
        main_logger.info("   GET  /api-info       - API信息")
        main_logger.info("=" * 60)

    def run(self):
        """运行服务"""
        # 初始化服务
        self.initialize_services()

        # 创建Flask应用
        app = self.create_flask_app()

        # 打印启动信息
        self.print_startup_info()

        # 启动服务
        main_logger = self.loggers['main']
        main_logger.info(f"🌐 服务正在启动，监听 {self.args.host}:{self.args.port}...")

        if self.args.debug:
            app.run(
                host=self.args.host,
                port=self.args.port,
                debug=True,
                threaded=True
            )
        else:
            serve(
                app,
                host=self.args.host,
                port=self.args.port,
                threads=8
            )

    def stop(self):
        """停止服务"""
        main_logger = self.loggers['main'] if self.loggers else None

        if main_logger:
            main_logger.info("🛑 正在停止服务...")

        # 停止ASR服务
        if self.asr_service:
            try:
                self.asr_service.stop()
                if main_logger:
                    main_logger.info("✅ ASR服务已停止")
            except Exception as e:
                if main_logger:
                    main_logger.error(f"❌ ASR服务停止失败: {e}")

        # 停止TTS服务
        if self.tts_service:
            try:
                self.tts_service.stop()
                if main_logger:
                    main_logger.info("✅ TTS服务已停止")
            except Exception as e:
                if main_logger:
                    main_logger.error(f"❌ TTS服务停止失败: {e}")


def main():
    """主函数"""
    args = parse_args()

    # 检查至少启用一个服务
    if not args.enable_asr and not args.enable_tts:
        print("错误：至少需要启用一个服务（--enable-asr 或 --enable-tts）")
        sys.exit(1)

    # 创建并运行服务
    service = VoiceService(args)

    try:
        service.run()
    except KeyboardInterrupt:
        service.loggers['main'].info("🛑 用户中断，正在停止服务...")
        service.stop()
    except Exception as e:
        if service.loggers:
            service.loggers['main'].error(f"❌ 服务运行异常: {e}")
        else:
            print(f"❌ 服务运行异常: {e}")
        service.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
