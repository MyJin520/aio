import os
from flask import jsonify, send_file

from utlis.sse_helpers import clear_sse_queue


def handle_status(asr_instance):
    """处理状态请求"""
    try:
        status_data = {
            "status": "healthy",
            "model_loaded": asr_instance.model is not None,
            "audio_running": asr_instance.audio_stream is not None and hasattr(asr_instance.audio_stream,
                                                                               'active') and asr_instance.audio_stream.active,
            "service_running": not asr_instance.stop_event.is_set(),
            "recording_active": asr_instance.recording_active,
            "listen_mode": asr_instance.listen_mode,
            "model_path": asr_instance.config.model_path,
            "sse_queue_size": asr_instance.sse_queue.qsize()
        }

        if not (status_data["model_loaded"] and status_data["audio_running"] and status_data["service_running"]):
            status_data["status"] = "unhealthy"

        return jsonify(status_data), 200 if status_data["status"] == "healthy" else 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def handle_listen(asr_instance, logger):
    """处理Listen模式请求"""
    try:
        asr_instance.start_listen_mode()
        return jsonify({
            "message": f"Listen模式已启动，结束关键词: {asr_instance.config.stop_keyword}，静音超时: {asr_instance.config.silence_timeout_seconds}s",
            "status": "success"
        }), 200
    except RuntimeError as e:
        return jsonify({"error": str(e), "status": "error"}), 400
    except Exception as e:
        logger.error(f"❌ Listen接口错误: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


def handle_clear_sse(asr_instance, logger):
    """处理SSE队列清空请求"""
    try:
        clear_sse_queue(asr_instance.sse_queue, logger)
        return jsonify({"message": "SSE队列已清空", "status": "success"}), 200
    except Exception as e:
        logger.error(f"❌ 清空SSE队列错误: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


def handle_send_audio(logger, audio_path="tmp.mp3"):
    """
    极简版：发送音频文件二进制流，无缓存、保留基础错误处理
    """
    try:
        # 基础文件校验
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return jsonify({"error": "音频文件不存在"}), 404

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"音频文件为空: {audio_path}")
            return jsonify({"error": "音频文件为空"}), 400

        # 发送二进制流，关闭所有缓存相关配置
        response = send_file(
            audio_path,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='recording.mp3',
            conditional=True,
            max_age=0,
        )

        # 强制禁用所有缓存
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        logger.info(f"音频流已发送: {audio_path}")
        return response, 200

    except Exception as e:
        logger.error(f"发送音频失败: {str(e)}")
        return jsonify({"error": f"发送音频失败: {str(e)}"}), 500
