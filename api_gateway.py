# ==================== BERT中文文本分类API网关 ====================
# 微服务架构中的API网关服务
# 负责：文本预处理、调用推理服务、结果后处理
# 采用HTTP转发模式，避免本地加载SavedModel的兼容性问题

import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from transformers import BertTokenizer

# --- 初始化 Flask 应用 ---
# 创建Flask应用实例，用于处理HTTP请求
app = Flask(__name__)

# --- 配置与全局变量 ---
# 从环境变量获取TensorFlow Serving的URL
# 默认值使用服务名'inference-service'，Docker Compose会自动解析
# 这是微服务架构中服务发现的关键配置
TF_SERVING_URL = os.environ.get('TF_SERVING_URL', 'http://inference-service:8501/v1/models/bert-chinese:predict')

# 分词器目录路径：包含BERT分词器和词汇表
TOKENIZER_DIR = 'saved_model/bert-chinese'

# 标签映射文件路径：将数字ID映射到文本标签
LABEL_MAP_PATH = os.path.join(TOKENIZER_DIR, 'label2id.txt')

# --- 在应用启动时加载分词器和标签 ---
# 这是API网关的核心初始化过程
# 加载分词器用于文本预处理，加载标签映射用于结果后处理
try:
    # 加载BERT分词器：用于将中文文本转换为模型输入格式
    # from_pretrained会自动下载或加载本地的分词器文件
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # 加载标签映射：将数字类别ID映射到可读的文本标签
    # 格式：每行包含"标签名\t数字ID"
    id2label = {}
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            label, idx = line.strip().split('\t')
            id2label[int(idx)] = label
            
except Exception as e:
    # 如果加载失败，记录错误并设置标志
    # 这样可以在运行时检查服务是否可用
    app.logger.error(f"启动错误: 加载分词器或标签文件时失败 - {e}")
    tokenizer = None
    id2label = None

# --- 定义预测路由 ---
# Flask路由装饰器，处理POST请求到/predict路径
# 这是API网关的主要入口点
@app.route("/predict", methods=['POST'])
def predict():
    # 服务可用性检查：确保分词器和标签映射已正确加载
    if tokenizer is None or id2label is None:
        return jsonify({"error": "分词器或标签文件未能成功加载，服务无法使用"}), 503

    # 请求格式验证：确保请求体是JSON格式且包含text字段
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "请求体必须是包含'text'字段的JSON"}), 400
    
    # 文本内容验证：确保text字段是有效的非空字符串
    text = request.json['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text'字段必须是有效的非空字符串"}), 400

    try:
        # ==================== 文本预处理阶段 ====================
        # 使用BERT分词器将原始文本转换为模型输入格式
        tokenized_inputs = tokenizer(
            [text],                    # 输入文本列表（这里只有一个文本）
            padding=True,              # 自动填充到相同长度
            truncation=True,           # 自动截断超长文本
            max_length=128,            # 最大序列长度（BERT标准）
            return_tensors="np"        # 返回numpy数组，便于序列化
        )
        
        # ==================== 构造推理请求 ====================
        # 构造TensorFlow Serving期望的请求格式
        # TensorFlow Serving REST API格式：{"instances": [{"input_ids": [...], ...}]}
        payload = {
            "instances": [
                {
                    "input_ids": tokenized_inputs['input_ids'][0].tolist(),           # 词汇ID序列
                    "attention_mask": tokenized_inputs['attention_mask'][0].tolist(), # 注意力掩码
                    "token_type_ids": tokenized_inputs['token_type_ids'][0].tolist() # 句子类型ID
                }
            ]
        }
        
        # ==================== 调用推理服务 ====================
        # 通过HTTP POST请求调用TensorFlow Serving推理服务
        # 这是微服务架构中服务间通信的核心
        resp = requests.post(TF_SERVING_URL, json=payload)
        
        # 检查推理服务响应状态
        if resp.status_code != 200:
            return jsonify({"error": f"推理服务调用失败: {resp.text}"}), 500
            
        # 解析推理服务返回的JSON结果
        result = resp.json()
        
        # ==================== 结果后处理阶段 ====================
        # 从推理结果中提取预测信息
        pred = result['predictions'][0]  # 获取第一个预测结果
        
        # 确定预测的类别ID
        # 如果推理服务直接返回class_id，使用它；否则从概率分布中找出最大值
        class_id = int(pred['class_id']) if 'class_id' in pred else int(np.argmax(pred['probabilities']))
        
        # 获取概率分布
        probabilities = pred['probabilities']
        
        # 将数字ID映射到文本标签
        predicted_label = id2label.get(class_id, "未知标签")
        
        # 获取预测置信度
        confidence = float(probabilities[class_id])
        
        # ==================== 构造最终响应 ====================
        # 返回用户友好的JSON响应
        return jsonify({
            "input_text": text,           # 原始输入文本
            "predicted_label": predicted_label,  # 预测的文本标签
            "confidence": confidence,     # 预测置信度（0-1）
            "class_id": class_id         # 预测的数字ID
        })
        
    except Exception as e:
        # 异常处理：记录错误并返回友好的错误信息
        app.logger.error(f"处理请求时发生未知错误: {e}")
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

@app.route("/predict_batch", methods=['POST'])
def predict_batch():
    if tokenizer is None or id2label is None:
        return jsonify({"error": "分词器或标签文件未能成功加载，服务无法使用"}), 503

    if not request.json or 'texts' not in request.json:
        return jsonify({"error": "请求体必须是包含'texts'字段的JSON，且为字符串列表"}), 400

    texts = request.json['texts']
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        return jsonify({"error": "'texts'字段必须是非空字符串列表"}), 400

    try:
        tokenized_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        payload = {
            "instances": [
                {
                    "input_ids": tokenized_inputs['input_ids'][i].tolist(),
                    "attention_mask": tokenized_inputs['attention_mask'][i].tolist(),
                    "token_type_ids": tokenized_inputs['token_type_ids'][i].tolist()
                } for i in range(len(texts))
            ]
        }
        resp = requests.post(TF_SERVING_URL, json=payload)
        if resp.status_code != 200:
            return jsonify({"error": f"推理服务调用失败: {resp.text}"}), 500
        result = resp.json()
        predictions = result['predictions']
        outputs = []
        for i, pred in enumerate(predictions):
            class_id = int(pred['class_id']) if 'class_id' in pred else int(np.argmax(pred['probabilities']))
            probabilities = pred['probabilities']
            predicted_label = id2label.get(class_id, "未知标签")
            confidence = float(probabilities[class_id])
            outputs.append({
                "input_text": texts[i],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "class_id": class_id
            })
        return jsonify({"results": outputs})
    except Exception as e:
        app.logger.error(f"批量处理请求时发生未知错误: {e}")
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

# --- 应用启动入口 ---
# 当直接运行此脚本时（非Docker环境），使用Flask内置服务器
# 在Docker环境中，会使用Gunicorn作为WSGI服务器
if __name__ == '__main__':
    # 开发模式启动：绑定到所有网络接口，启用调试模式
    app.run(host='0.0.0.0', port=5001, debug=True)

# ==================== 微服务架构说明 ====================
# 1. 职责分离：
#    - API网关：文本预处理、结果后处理、用户接口
#    - 推理服务：模型计算、高性能推理
#
# 2. HTTP转发模式优势：
#    - 避免本地加载SavedModel的兼容性问题
#    - 符合微服务架构的最佳实践
#    - 便于独立扩展和维护
#
# 3. 服务发现：
#    - 通过环境变量配置推理服务地址
#    - Docker Compose自动处理DNS解析
#    - 支持服务名访问（inference-service）
#
# 4. 错误处理：
#    - 分层错误处理：请求验证、服务调用、结果处理
#    - 友好的错误信息返回
#    - 详细的错误日志记录 