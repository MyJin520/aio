from flask import Flask, request, jsonify, send_file, after_this_request
import threading
import uuid
import time
import json
import os
from datetime import datetime
from itertools import zip_longest

from service.tts_engine import TTSServer


def log_request(func):
    """请求日志装饰器，消除冗余日志代码"""

    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            response, status_code = func(self, *args, **kwargs)
            response_time = time.time() - start_time
            self._log_access(
                endpoint=request.path,
                method=request.method,
                status_code=status_code,
                processing_time=response_time,
                request_data=getattr(wrapper, 'request_data', None),
                response_data=getattr(wrapper, 'response_data', None)
            )
            return response, status_code
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            self.tts_server.error_logger.error(f"请求处理异常: {error_msg}")
            self._log_access(
                endpoint=request.path,
                method=request.method,
                status_code=500,
                processing_time=response_time,
                response_data={"error": error_msg}
            )
            return jsonify({"error": error_msg}), 500

    return wrapper


class APIServer:
    def __init__(self, tts_server: TTSServer):
        self.app = Flask(__name__)
        self.app.after_request(self.after_request)  # 全局CORS处理
        self.tts_server = tts_server
        self.engine_lock = threading.Lock()  # 保持线程安全
        self._setup_routes()

    def after_request(self, response):
        """全局CORS配置，无需在OPTIONS中重复设置"""
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization,Accept,X-Requested-With',
            'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS',
            'Access-Control-Expose-Headers': 'Content-Disposition'
        })
        return response

    def _log_access(self, endpoint, method, status_code, processing_time, request_data=None, response_data=None):
        """统一日志格式"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'processing_time': f"{processing_time:.3f}s",
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
        }
        if request_data:
            log_entry['request_data'] = request_data
        if response_data:
            log_entry['response_data'] = response_data
        self.tts_server.access_logger.info(json.dumps(log_entry, ensure_ascii=False))

    def _setup_routes(self):
        @self.app.route('/tts/create', methods=['POST', 'OPTIONS'])
        def tts_create():
            if request.method == 'OPTIONS':
                return jsonify({'status': 'ok'}), 200

            request_id = str(uuid.uuid4())[:8]
            start_time = time.time()
            temp_file_path = None

            if not self.tts_server.is_initialized:
                response = jsonify({
                    "error": "服务未就绪",
                    "details": self.tts_server.initialization_error,
                    "request_id": request_id
                })
                return response, 503

            try:

                text = request.form.get('text', '').strip()
                if not text:
                    response = jsonify({
                        "error": "缺少text参数",
                        "request_id": request_id
                    })
                    return response, 400

                ref_audio_files = request.files.getlist('ref_audio')
                ref_texts = request.form.getlist('ref_text')
                refs_data = [
                    {
                        "audio_data": file.read(),
                        "text": text or ""
                    }
                    for file, text in zip_longest(ref_audio_files, ref_texts)
                    if file and file.filename
                ]

                request_data = {
                    "text_length": len(text),
                    "reference_count": len(refs_data),
                    "text_preview": text[:50] + "..." if len(text) > 50 else text
                }

                with self.engine_lock:

                    temp_file_path, processing_time, audio_size_kb = self.tts_server.generate_tts(
                        text, refs_data, request_id=request_id
                    )

                @after_this_request
                def cleanup(response):
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            self.tts_server.error_logger.error(f"删除临时文件失败: {str(e)}")
                    return response

                response_data = {
                    "audio_size_kb": f"{audio_size_kb:.1f}",
                    "processing_time": f"{processing_time:.2f}s",
                    "total_time": f"{time.time() - start_time:.2f}s"
                }

                return send_file(
                    temp_file_path,
                    mimetype='audio/mpeg',
                    as_attachment=False,
                    download_name='tts_output.mp3',
                    conditional=True
                ), 200

            except Exception as e:
                error_msg = f"处理失败: {str(e)}"
                self.tts_server.error_logger.error(f"TTS创建失败 (request_id: {request_id}): {error_msg}")
                return jsonify({
                    "error": error_msg,
                    "request_id": request_id
                }), 500

        @self.app.route('/tts/status', methods=['GET', 'OPTIONS'])
        def tts_status():
            if request.method == 'OPTIONS':
                return jsonify({'status': 'ok'}), 200

            status_info, status_code = self.tts_server.get_status()
            return jsonify(status_info), status_code

    def run(self, host, port):
        """启动服务，优化服务器配置"""
        self.tts_server.logger.info(f"启动API服务: http://{host}:{port}")
        self.tts_server.logger.info(f"可用端点:")
        self.tts_server.logger.info(f"  - POST /tts/create : 创建TTS音频 (MP3格式)")
        self.tts_server.logger.info(f"  - GET  /tts/status : 获取服务状态")
        self.tts_server.logger.info(f"日志目录: {self.tts_server.config.log_dir}")

        from werkzeug.serving import WSGIRequestHandler
        import socketserver
        socketserver.TCPServer.allow_reuse_address = True
        WSGIRequestHandler.timeout = 300

        self.app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            request_handler=WSGIRequestHandler
        )
