from flask import Flask, Response, jsonify

from api.asr_api import ASRHandlers
from api.tts_api import TTSHandlers
from services.asr import ASRService
from services.tts import TTSService
from utils.cors import CORSManager
from utils.logger import get_logger
from utils.sse import SSEHelper


class VoiceServiceRouter:
    """语音服务路由器"""

    def __init__(self, app: Flask, asr_service: ASRService = None,
                 tts_service: TTSService = None, logger=None):
        self.app = app
        self.asr_service = asr_service
        self.tts_service = tts_service
        self.logger = logger or get_logger("router")

        # 初始化处理器
        self.asr_handlers = ASRHandlers(asr_service, logger) if asr_service else None
        self.tts_handlers = TTSHandlers(tts_service, logger) if tts_service else None

        # 设置CORS
        CORSManager.setup_cors(app)

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册所有路由"""

        # ASR路由
        if self.asr_handlers:
            @self.app.route('/asr/status', methods=['GET'])
            def asr_status():
                return self.asr_handlers.handle_status()

            @self.app.route('/asr/listen', methods=['POST'])
            def asr_listen():
                return self.asr_handlers.handle_listen()

            @self.app.route('/asr/stream', methods=['GET'])
            def asr_stream():
                return Response(
                    SSEHelper.generate_sse_events(self.asr_service, self.logger),
                    mimetype='text/event-stream'
                )

            @self.app.route('/asr/clear-sse', methods=['POST'])
            def asr_clear_sse():
                return self.asr_handlers.handle_clear_sse()

            @self.app.route('/asr/audio', methods=['GET'])
            def asr_audio():
                return self.asr_handlers.handle_send_audio()

        # TTS路由
        if self.tts_handlers:
            @self.app.route('/tts/create', methods=['POST', 'OPTIONS'])
            def tts_create():
                return self.tts_handlers.handle_create()

            @self.app.route('/tts/status', methods=['GET', 'OPTIONS'])
            def tts_status():
                return self.tts_handlers.handle_status()

        # 健康检查路由
        @self.app.route('/health', methods=['GET'])
        def health():
            status = {"status": "healthy", "services": {}}

            if self.asr_service:
                status["services"]["asr"] = self.asr_service.get_status()

            if self.tts_service:
                status["services"]["tts"] = self.tts_service.get_status()

            return jsonify(status), 200

        @self.app.route('/api-info', methods=['GET'])
        def api_info():
            """API信息"""
            endpoints = []

            if self.asr_handlers:
                endpoints.extend([
                    {"path": "/asr/status", "method": "GET", "description": "ASR服务状态"},
                    {"path": "/asr/listen", "method": "POST", "description": "启动Listen模式"},
                    {"path": "/asr/stream", "method": "GET", "description": "实时SSE流"},
                    {"path": "/asr/clear-sse", "method": "POST", "description": "清空SSE队列"},
                    {"path": "/asr/audio", "method": "GET", "description": "获取音频文件"},
                ])

            if self.tts_handlers:
                endpoints.extend([
                    {"path": "/tts/create", "method": "POST", "description": "生成语音"},
                    {"path": "/tts/status", "method": "GET", "description": "TTS服务状态"},
                ])

            endpoints.extend([
                {"path": "/health", "method": "GET", "description": "服务健康检查"},
                {"path": "/api-info", "method": "GET", "description": "API信息"},
            ])

            return jsonify({
                "service": "AI Voice Service",
                "version": "1.0.0",
                "endpoints": endpoints
            }), 200