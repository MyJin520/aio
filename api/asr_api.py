import os
from flask import jsonify, send_file

from services.asr import ASRService
from utils.sse import SSEHelper


class ASRHandlers:
    """ASR API处理器"""

    def __init__(self, asr_service: ASRService, logger):
        self.asr_service = asr_service
        self.logger = logger

    def handle_status(self):
        """处理状态请求"""
        try:
            status_data = self.asr_service.get_status()
            status_data["status"] = "healthy" if self.asr_service.is_running else "unhealthy"
            return jsonify(status_data), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    def handle_listen(self):
        """处理Listen模式请求"""
        try:
            if self.asr_service.listen_mode:
                return jsonify({"error": "已在Listen模式中", "status": "error"}), 400

            # 先重置状态，但不设置listen_mode为True
            self.asr_service._reset_recognition_state()

            # 然后设置listen_mode为True，这样就不会被_reset_recognition_state重置
            self.asr_service.listen_mode = True
            self.asr_service.listen_results.clear()
            self.asr_service.recording_active = False

            return jsonify({
                "message": f"Listen模式已启动，结束关键词: {self.asr_service.config.stop_keyword}",
                "status": "success"
            }), 200

        except Exception as e:
            self.logger.error(f"❌ Listen接口错误: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    def handle_clear_sse(self):
        """处理SSE队列清空请求"""
        try:
            SSEHelper.clear_sse_queue(self.asr_service.sse_queue, self.logger)
            return jsonify({"message": "SSE队列已清空", "status": "success"}), 200
        except Exception as e:
            self.logger.error(f"❌ 清空SSE队列错误: {str(e)}")
            return jsonify({"error": str(e), "status": "error"}), 500

    def handle_send_audio(self, audio_path="tmp.mp3"):
        """发送音频文件"""
        try:
            if not os.path.exists(audio_path):
                self.logger.error(f"音频文件不存在: {audio_path}")
                return jsonify({"error": "音频文件不存在"}), 404

            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                self.logger.error(f"音频文件为空: {audio_path}")
                return jsonify({"error": "音频文件为空"}), 400

            response = send_file(
                audio_path,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='recording.mp3',
                conditional=True,
                max_age=0,
            )

            # 禁用缓存
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            self.logger.info(f"音频流已发送: {audio_path}")
            return response, 200

        except Exception as e:
            self.logger.error(f"发送音频失败: {str(e)}")
            return jsonify({"error": f"发送音频失败: {str(e)}"}), 500