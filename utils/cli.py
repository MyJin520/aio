import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI语音服务（ASR + TTS）')

    # 服务配置
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    parser.add_argument('--ignore-errors', action='store_true',
                        help='忽略服务初始化错误')

    # ASR配置
    parser.add_argument('--enable-asr', action='store_true',
                        help='启用ASR服务')
    parser.add_argument('--asr-model-path', type=str,
                        help='ASR模型路径')
    parser.add_argument('--start-keyword', type=str, default='开始',
                        help='开始关键词')
    parser.add_argument('--stop-keyword', type=str, default='结束',
                        help='结束关键词')
    parser.add_argument('--silence-threshold', type=float, default=0.007,
                        help='静音阈值')
    parser.add_argument('--silence-timeout', type=float, default=7.0,
                        help='静音超时秒数')

    # TTS配置
    parser.add_argument('--enable-tts', action='store_true',
                        help='启用TTS服务')
    parser.add_argument('--tts-port', type=int,
                        help='TTS服务端口（默认与主端口相同）')
    parser.add_argument('--tts-model-path', type=str,
                        help='TTS模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='运行设备')
    parser.add_argument('--compile', action='store_true',
                        help='启用模型编译优化')

    return parser.parse_args()
