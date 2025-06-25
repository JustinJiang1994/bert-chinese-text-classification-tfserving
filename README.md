# 中文文本分类（BERT + TensorFlow Serving）

## 项目简介
本项目旨在基于BERT模型进行中文文本分类，最终通过TensorFlow Serving部署模型服务。

## 项目流程

### 1. 环境准备
本项目依赖 `TensorFlow` 和 `Transformers` 等库，建议使用 `pip3` 安装。

```bash
pip3 install -r requirements.txt
```

### 2. 数据集切分
- 原始数据位于 `data/initial.data`，包含 `label` 和 `txt` 两列，以Tab分隔。
- 运行 `split_data.py` 脚本可将其按 8:1:1 的比例切分为训练、验证和测试集。

```bash
# 此脚本依赖 pandas 和 scikit-learn
python3 scripts/split_data.py
```
切分后的文件（`train.data`, `val.data`, `test.data`）将保存在 `data/` 目录下。

### 3. 模型训练
- 使用 `train_bert.py` 脚本进行模型训练。
- 该脚本会加载 `bert-base-chinese` 预训练模型，并在我们的数据集上进行微调。
- 训练完成后，模型将保存在 `saved_model/bert-chinese/` 目录下，并在测试集上输出准确率。

```bash
python3 train_bert.py
```

### 4. 模型导出
- 为了让 TensorFlow Serving 能够加载模型，需要将其转换为标准的 `SavedModel` 格式。
- `export_model.py` 脚本负责此项转换，并为模型定义一个接受原始文本输入的推理接口。

```bash
python3 export_model.py
```
- 导出的模型位于 `tf_serving_model/bert-chinese/1`，其中 `1` 是模型版本号。

### 5. 服务部署
推荐使用 Docker 启动 TensorFlow Serving，这样可以避免复杂的环境依赖问题。
```bash
# 确保Docker已启动
# --name 为容器命名
# -p 将本机的8501端口映射到容器的8501(REST API)
# -v 将本地模型目录挂载到容器的/models目录
# -e 指定要加载的模型名称
docker run -d --name tf-bert-serving -p 8501:8501 \
  -v "$(pwd)/tf_serving_model:/models" \
  -e MODEL_NAME=bert-chinese \
  tensorflow/serving
```
> **为什么要用 Docker？**
> 直接在宿主机上通过 `tensorflow_model_server` 命令启动服务理论上可行，但通常会遇到C++库依赖、版本不匹配等复杂的环境问题。使用官方提供的 `tensorflow/serving` Docker 镜像有以下好处：
> - **环境隔离**: 无需在本地安装任何 TensorFlow Serving 相关的依赖，所有运行时都在容器内，与宿主机环境完全隔离。
> - **版本一致性**: 确保了开发、测试和生产环境的一致性，避免了“在我机器上能跑”的问题。
> - **部署便捷**: 一行命令即可启动、停止和管理服务，极大简化了部署流程。

### 6. 调用服务
`predict_client.py` 脚本演示了如何调用已部署的服务。它会在本地加载分词器，对文本进行预处理，然后向服务发送请求。
```bash
pip3 install requests
python3 predict_client.py
```
成功调用后，会输出预测结果：
```
模型预测结果:
==============================
文本: '这手机拍照真好看，我很喜欢！'
  -> 预测类别: 1 (置信度: 0.9945)
------------------------------
文本: '电池太不耐用了，一天要充好几次电。'
  -> 预测类别: -1 (置信度: 0.9620)
------------------------------
文本: '手机屏幕显示效果还行，中规中矩。'
  -> 预测类别: 1 (置信度: 0.8104)
------------------------------
```

## 目录结构
```
bert-chinese-text-classification-tfserving/
  ├── data/
  │   ├── initial.data
  │   ├── train.data
  │   ├── val.data
  │   └── test.data
  ├── scripts/
  │   └── split_data.py
  ├── saved_model/
  │   └── bert-chinese/      # 训练后保存的Hugging Face模型
  ├── tf_serving_model/
  │   └── bert-chinese/
  │       └── 1/             # 导出的TF Serving模型
  ├── train_bert.py
  ├── export_model.py
  ├── predict_client.py
  └── requirements.txt
```
## 常见问题与解决方法 (Troubleshooting)

在开发过程中，我们遇到了一些典型的环境和部署问题，这里记录最终的解决方案。

### 1. 问题：依赖库版本冲突
- **现象**: `ImportError`, `ValueError: numpy.dtype size changed` 等。
- **原因**: `tensorflow`, `tf_keras`, `numpy`, `pandas` 等库之间存在严格的版本依赖。`pip` 在安装或升级某个库时，可能会引入与其他库不兼容的版本。
- **解决方案**:
  1.  当 `tf.keras` 无法导入时，根据提示安装 `tf_keras` (`pip install tf_keras`) 并修改代码。
  2.  当出现 `numpy` 不兼容问题时，通常是 `pandas` 或其他库需要升级 (`pip install --upgrade pandas`) 以匹配新版 `numpy`。
  3.  **最佳实践**: 维护一个稳定的 `requirements.txt` 文件，在新环境中一次性安装所有验证过的依赖版本。

### 2. 问题：模型导出失败或服务无法调用
- **现象**:
  - `export_model.py` 报错 `ValueError` 或 `TypeError`。
  - TensorFlow Serving 返回 `400 Bad Request`，错误信息为 `No OpKernel was registered to support Op 'EagerPyFunc'`。
- **原因**: 这是部署中最核心的问题。`tf.py_function` (以及任何纯Python逻辑，如`tokenizer`) 无法被标准的 TensorFlow Serving C++ 后端执行。我们为了修复导出错误而引入的 `tf.py_function`，恰恰是服务无法识别的操作。
- **解决方案 (标准部署架构)**:
  1.  **服务器只负责纯计算**：将所有预处理/后处理逻辑（如分词）从模型导出代码中完全剥离。导出的 `SavedModel` 应该是一个纯净的计算图，其接口只接受已经处理好的张量（如 `input_ids`）作为输入。
  2.  **客户端负责所有预处理**: 在调用服务的客户端 (`predict_client.py`) 中加载 `tokenizer`，完成从**原始文本 -> 分词 -> 张量**的所有转换工作。然后将这些张量作为请求体发送给 TensorFlow Serving。
  - *这种 "客户端分词" 的架构是业界标准，它不仅解决了技术问题，也使得模型服务本身更轻量、更高效。*

