# ASR & TTS 整合服务项目文档
# ASR & TTS Integrated Service Project Documentation

## 项目介绍
## Project Introduction
> 将单独的asr服务和tts服务整合在一起，减少单独打包的资源浪费
> Integrate independent ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) services to reduce resource waste caused by separate packaging.

## 项目启动
## Project Startup
> 启动asr服务和tts服务并使用编译加速，更多命令行参数参考cli.py
> Start ASR and TTS services with compilation acceleration. For more command-line parameters, refer to cli.py
> python main.py --enable-asr --enable-tts --compile

## 核心依赖
## Core Dependencies
> pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> pip install fish_speech
> pip install funasr
> pip install flask
> pip install waitress
> pip install sounddevice

## 基础启动镜像
## Basic Startup Image
【会默认启动asr和tts服务及编译加速】
【Will start ASR and TTS services with compilation acceleration by default】

> 模型:
> Models:
> ASR模型：来自funasr模型列表
> ASR Model: Selected from the funasr model list
> TTS模型：来自fish_speech模型列表
> TTS Model: Selected from the fish_speech model list
> docker run --gpus all -d -p 5000:5000 --name aio_local -v 本地tts模型目录:/app/tts_model -v 本地asr模型目录:/app/asr_model 镜像标识
> docker run --gpus all -d -p 5000:5000 --name aio_local -v local_tts_model_directory:/app/tts_model -v local_asr_model_directory:/app/asr_model image_tag

## 结语
## Conclusion
> 如果你觉得该项目对你有帮助,欢迎给个star,如果有任何问题，欢迎提交issue,或者对项目有任何建议，期待你的pr😉
> If you find this project helpful, please feel free to give it a star. If you encounter any issues, welcome to submit an issue. If you have any suggestions for the project, we are looking forward to your pull request 😉
