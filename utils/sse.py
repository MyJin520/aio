import json
import queue
import time



class SSEHelper:
    """SSE事件流助手"""

    @staticmethod
    def send_sse_data(
            sse_queue: queue.Queue,
            data_type: str,
            text: str,
            maxsize: int = 100,
            **kwargs
    ) -> None:
        """发送SSE数据"""
        sse_data = {
            'type': data_type,
            'text': text,
            'timestamp': time.time(),
            **kwargs
        }
        try:
            sse_queue.put_nowait(json.dumps(sse_data))
        except queue.Full:
            try:
                sse_queue.get_nowait()
                sse_queue.put_nowait(json.dumps(sse_data))
            except queue.Empty:
                pass

    @staticmethod
    def clear_sse_queue(sse_queue: queue.Queue, logger) -> None:
        """清空SSE队列"""
        try:
            while not sse_queue.empty():
                sse_queue.get_nowait()
            logger.debug("SSE队列已清空")
        except Exception as e:
            logger.warning(f"清空SSE队列失败: {str(e)}")

    @staticmethod
    def generate_sse_events(asr_instance, logger):
        """生成SSE事件流"""
        try:
            while not asr_instance.stop_event.is_set():
                try:
                    msg = asr_instance.sse_queue.get(timeout=1.0)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            logger.info("客户端断开SSE连接")
            SSEHelper.clear_sse_queue(asr_instance.sse_queue, logger)
        except Exception as e:
            logger.error(f"SSE流错误: {str(e)}")