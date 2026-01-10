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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 创建虚拟环境
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 设置pip源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖并清理缓存
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir fish_speech \
    && pip install --no-cache-dir funasr \
    && pip install --no-cache-dir flask waitress pydub sounddevice \
    && rm -rf /root/.cache/pip

# 第二阶段：运行时环境
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.11.8-slim AS runtime

WORKDIR /app

# 设置国内源 - 直接写入
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-backports main non-free contrib" >> /etc/apt/sources.list

# 只安装运行时必要的最小依赖（必须包含gcc!）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg portaudio19-dev libportaudio2 libportaudiocpp0 \
    gcc g++ libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 复制虚拟环境和项目文件
COPY --from=builder /app/venv /app/venv
COPY . /app/

ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache/torch \
    # 设置编译器环境变量
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# 创建目录
RUN mkdir -p /app/logs /app/asr_model /app/tts_model /app/.cache/torch

EXPOSE 5000

CMD ["python", "main.py", "--enable-tts", "--compile"]
