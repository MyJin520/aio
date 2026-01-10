import argparse


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FunASR 实时语音识别服务')
    parser.add_argument('--port', type=int, default=5000, help='Flask端口（默认5000）')
    parser.add_argument('--start-keyword', type=str, default='开始', help='开始关键词')
    parser.add_argument('--stop-keyword', type=str, default='结束', help='结束关键词')
    parser.add_argument('--silence-threshold', type=float, default=0.001, help='静音阈值')
    parser.add_argument('--silence-timeout', type=float, default=7.0, help='静音超时秒数')
    parser.add_argument('--model-path', type=str, default=None, help='本地模型路径（默认为当前目录下的models目录）')
    parser.add_argument('--model-name', type=str, default='paraformer-zh-streaming', help='模型名称')
    parser.add_argument('--model-revision', type=str, default='v2.0.4', help='模型版本')
    return parser.parse_args()