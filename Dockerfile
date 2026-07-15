# ============================================
# 第一阶段：构建依赖层
# 使用更小的镜像减少体积
# ============================================
FROM python:3.13-slim-bookworm AS builder

# 安装系统依赖（opencv 和 OCR 模型需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制 requirements，利用 Docker 缓存
COPY requirements.txt .

# 安装 Python 依赖到临时目录
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# 第二阶段：运行镜像
# ============================================
FROM python:3.13-slim-bookworm

# 安装运行时必需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码和 JSON 注册表文件
COPY main.py .

# 创建数据目录（运行时挂载）
RUN mkdir -p uploads chroma_db && \
    touch pdf_registry.json jd_registry.json resume_registry.json document_registry.json interview_sessions.json

# 声明端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
