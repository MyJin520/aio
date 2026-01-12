# 第一阶段：构建环境
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11.8-slim AS builder

WORKDIR /app

# 设置国内源 - 直接写入
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-backports main non-free contrib" >> /etc/apt/sources.list

# 一次性安装所有依赖并清理
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake build-essential \
    libffi-dev libssl-dev \
    ffmpeg portaudio19-dev libportaudio2 libportaudiocpp0 \
    pulseaudio alsa-utils libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 创建虚拟环境
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 设置pip源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖并清理缓存，补充缺失的python音频依赖
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir fish_speech \
    && pip install --no-cache-dir funasr \
    && pip install --no-cache-dir flask waitress pydub sounddevice triton \
    && pip install --no-cache-dir soundfile==0.13.1 PyAudio==0.2.14 \
    && rm -rf /root/.cache/pip

# 第二阶段：运行时环境
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11.8-slim AS runtime

WORKDIR /app

# 设置国内源 - 直接写入
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-backports main non-free contrib" >> /etc/apt/sources.list

# 只安装运行时必要的最小依赖，补充缺失的音频核心依赖 pulseaudio alsa-utils libsndfile1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg portaudio19-dev libportaudio2 libportaudiocpp0 \
    gcc g++ libgomp1 \
    pulseaudio alsa-utils libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ===================== 原始镜像最核心缺失的这几行 =====================
# 修复音频设备检测失败的关键：配置PulseAudio虚拟音频驱动
RUN mkdir -p /etc/pulse && \
    echo "default-server = unix:/tmp/pulseaudio.socket" > /etc/pulse/client.conf && \
    echo "autospawn = no" >> /etc/pulse/client.conf && \
    echo "daemon-binary = /bin/true" >> /etc/pulse/client.conf && \
    echo "enable-shm = no" >> /etc/pulse/client.conf
# =====================================================================

# 复制虚拟环境和项目文件
COPY --from=builder /app/venv /app/venv
COPY . /app/

# 环境变量，补充缺失的音频相关环境变量
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.cache/torch \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PULSE_SERVER=host.docker.internal \
    PULSE_COOKIE=/tmp/pulse_cookie

# 创建目录
RUN mkdir -p /app/logs /app/asr_model /app/tts_model /app/.cache/torch

EXPOSE 5000

# 保留你的原始启动命令，无需修改
CMD ["python", "main.py", "--enable-asr", "--enable-tts", "--compile"]
