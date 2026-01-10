from flask import Flask, Response

from api.asr_handlers import handle_status, handle_listen, handle_clear_sse, handle_send_audio
from utlis.sse_helpers import generate_sse_events


def register_routes(app: Flask, asr_instance, logger):
    """注册所有API路由"""

    @app.route('/asr/status', methods=['GET'])
    def status():
        return handle_status(asr_instance)

    @app.route('/asr/listen', methods=['POST'])
    def listen():
        return handle_listen(asr_instance, logger)

    @app.route('/asr/stream', methods=['GET'])
    def stream():
        return Response(
            generate_sse_events(asr_instance, logger),
            mimetype='text/event-stream'
        )

    @app.route('/asr/clear-sse', methods=['POST'])
    def clear_sse():
        return handle_clear_sse(asr_instance, logger)

    @app.route('/asr/audio', methods=['GET'])
    def send_audio():
        return handle_send_audio(logger=logger)
