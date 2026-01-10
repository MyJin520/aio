import json
import uuid
import time
from datetime import datetime
from flask import request, jsonify, send_file
from itertools import zip_longest

from services.tts import TTSService


class TTSHandlers:
    """TTS API处理器"""

    def __init__(self, tts_service: TTSService, logger):
        self.tts_service = tts_service
        self.logger = logger

    def handle_create(self):
        """处理TTS创建请求"""
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if not self.tts_service.is_running:
            return jsonify({
                "error": "服务未就绪",
                "details": self.tts_service.initialization_error,
                "request_id": request_id
            }), 503

        try:
            text = request.form.get('text', '').strip()
            if not text:
                return jsonify({
                    "error": "缺少text参数",
                    "request_id": request_id
                }), 400

            # 处理参考音频
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

            # 生成语音
            audio_stream, processing_time, audio_size_kb = self.tts_service.generate_speech(
                text, refs_data, request_id
            )

            # 记录访问日志
            self._log_access('/tts/create', 'POST', 200, time.time() - start_time, {
                "text_length": len(text),
                "reference_count": len(refs_data)
            })

            return send_file(
                audio_stream,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='tts_output.mp3',
                conditional=True
            ), 200

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.logger.error(f"TTS创建失败 (request_id: {request_id}): {error_msg}")

            self._log_access('/tts/create', 'POST', 500, time.time() - start_time, {
                "error": error_msg
            })

            return jsonify({
                "error": error_msg,
                "request_id": request_id
            }), 500

    def handle_status(self):
        """处理TTS状态请求"""
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        status_info = self.tts_service.get_status()
        self._log_access('/tts/status', 'GET', 200, 0)

        return jsonify(status_info), 200

    def _log_access(self, endpoint, method, status_code, processing_time, data=None):
        """记录访问日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'processing_time': f"{processing_time:.3f}s",
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
        }

        if data:
            log_entry['data'] = data

        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
