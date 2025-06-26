# ==================== API网关容器构建文件 ====================
# 用于构建运行Flask API网关的Docker镜像
# 包含Python环境、依赖库、应用代码和启动配置

# 基础镜像：使用Python 3.8官方镜像
# python:3.8 是一个轻量级的Python运行环境
# 包含了Python解释器和基本的系统工具
FROM python:3.8

# 设置容器内的工作目录
# 后续的所有操作都会在这个目录下进行
# 这有助于保持容器内文件结构的整洁
WORKDIR /app

# 安装系统级依赖
# apt-get update：更新包管理器索引
# apt-get install -y：安装libhdf5-dev库（HDF5开发库）
# libhdf5-dev是TensorFlow等机器学习库的依赖项
# 用于处理HDF5格式的数据文件
RUN apt-get update && apt-get install -y libhdf5-dev

# 复制Python依赖文件
# 将requirements.txt复制到容器内的当前工作目录
# 这个文件列出了所有Python包的依赖关系
COPY requirements.txt .

# 安装Python依赖包
# --no-cache-dir：不缓存下载的包，减少镜像大小
# -r requirements.txt：从requirements.txt文件安装所有依赖
# && pip install flask gunicorn：额外安装Flask和Gunicorn
# Flask：Web框架，用于构建API
# Gunicorn：WSGI服务器，用于生产环境运行Flask应用
RUN pip install --no-cache-dir -r requirements.txt && pip install flask gunicorn

# 复制应用代码
# 将当前目录（宿主机）的所有文件复制到容器内
# 包括api_gateway.py、saved_model/等所有项目文件
# 注意：这里会复制整个项目目录，包括模型文件
COPY . .

# 暴露端口
# 声明容器将使用5001端口
# 这只是一个文档说明，实际端口映射在docker-compose.yml中配置
# 有助于其他开发者了解应用使用的端口
EXPOSE 5001

# 容器启动命令
# 使用Gunicorn作为WSGI服务器运行Flask应用
# --bind 0.0.0.0:5001：绑定到所有网络接口的5001端口
# api_gateway:app：指定Flask应用对象
# api_gateway是模块名，app是Flask应用实例
# 使用Gunicorn而不是Flask内置服务器，更适合生产环境
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "api_gateway:app"]

# ==================== 构建过程说明 ====================
# 1. 基础环境：从Python 3.8镜像开始
# 2. 系统依赖：安装HDF5库等系统级依赖
# 3. Python环境：安装所有Python包依赖
# 4. 应用代码：复制项目代码到容器
# 5. 端口配置：声明应用使用的端口
# 6. 启动命令：配置容器启动时执行的命令

# ==================== 优化建议 ====================
# 1. 多阶段构建：可以使用多阶段构建减少最终镜像大小
# 2. 依赖缓存：可以优化依赖安装顺序，利用Docker层缓存
# 3. 安全扫描：可以添加安全扫描步骤
# 4. 健康检查：可以添加HEALTHCHECK指令
# 5. 用户权限：可以创建非root用户运行应用 