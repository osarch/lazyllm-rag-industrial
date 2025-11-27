# 基础镜像：Python 3.10 + CUDA 11.7（LazyLLM推荐环境，兼容性最好）
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# 设置维护者信息（可选）
LABEL maintainer="AI实战工程师 <osarch@example.com>"
LABEL description="LazyLLM工业级RAG服务镜像（支持多模态+GPU加速）"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（基础工具+OCR依赖+中文支持）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tesseract-ocr \
    tesseract-ocr-chi-sim \  # 中文OCR语言包
    libgl1-mesa-glx \  # 图片处理依赖
    libglib2.0-0 \  # 图片处理依赖
    locales \  # 本地化支持
    && rm -rf /var/lib/apt/lists/* \
    # 配置中文 locale（避免中文乱码）
    && locale-gen zh_CN.UTF-8 \
    && update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \  # 不生成.pyc文件
    PYTHONUNBUFFERED=1 \  # 无缓冲输出（日志实时打印）
    PYTHONIOENCODING=utf-8 \  # 编码格式
    LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    TZ=Asia/Shanghai

# 安装Python依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    # 安装NVIDIA CUDA工具包（确保torch能识别GPU）
    && pip install --no-cache-dir nvidia-cudnn-cu11==8.9.2.26

# 复制项目文件（从项目根目录复制到容器/app目录）
COPY . /app/

# 创建日志目录并设置权限（避免权限不足）
RUN mkdir -p /app/logs && chmod 777 /app/logs \
    && mkdir -p /app/vector_db && chmod 777 /app/vector_db \
    && mkdir -p /app/multimodal_vector_db && chmod 777 /app/multimodal_vector_db

# 暴露端口（FastAPI服务端口）
EXPOSE 8000

# 启动命令（用sh执行，支持环境变量替换）
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers $WORKERS --log-config /app/api/logging.py"]
