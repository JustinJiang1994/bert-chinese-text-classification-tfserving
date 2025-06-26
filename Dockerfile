# 使用本地Python基础镜像
FROM python:3.8

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y libhdf5-dev

# 复制依赖文件并安装
COPY requirements.txt .
# 添加gunicorn用于生产环境运行
RUN pip install --no-cache-dir -r requirements.txt && pip install flask gunicorn

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5001

# 定义容器启动时运行的命令
# 使用Gunicorn作为WSGI服务器
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "api_gateway:app"] 