import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from transformers import BertTokenizer

# --- 初始化 Flask 应用 ---
app = Flask(__name__)

# --- 配置与全局变量 ---
TF_SERVING_URL = os.environ.get('TF_SERVING_URL', 'http://inference-service:8501/v1/models/bert-chinese:predict')
TOKENIZER_DIR = 'saved_model/bert-chinese'
LABEL_MAP_PATH = os.path.join(TOKENIZER_DIR, 'label2id.txt')

# --- 在应用启动时加载分词器和标签 ---
try:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
    id2label = {}
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            label, idx = line.strip().split('\t')
            id2label[int(idx)] = label
except Exception as e:
    app.logger.error(f"启动错误: 加载分词器或标签文件时失败 - {e}")
    tokenizer = None
    id2label = None

# --- 定义预测路由 ---
@app.route("/predict", methods=['POST'])
def predict():
    if tokenizer is None or id2label is None:
        return jsonify({"error": "分词器或标签文件未能成功加载，服务无法使用"}), 503

    if not request.json or 'text' not in request.json:
        return jsonify({"error": "请求体必须是包含'text'字段的JSON"}), 400
    
    text = request.json['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text'字段必须是有效的非空字符串"}), 400

    try:
        # 1. 文本预处理（分词）
        tokenized_inputs = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"  # 返回numpy数组，便于序列化
        )
        # 2. 构造TF Serving请求
        payload = {
            "instances": [
                {
                    "input_ids": tokenized_inputs['input_ids'][0].tolist(),
                    "attention_mask": tokenized_inputs['attention_mask'][0].tolist(),
                    "token_type_ids": tokenized_inputs['token_type_ids'][0].tolist()
                }
            ]
        }
        # 3. 发送请求到TF Serving
        resp = requests.post(TF_SERVING_URL, json=payload)
        if resp.status_code != 200:
            return jsonify({"error": f"推理服务调用失败: {resp.text}"}), 500
        result = resp.json()
        # 4. 解析返回结果
        # 假设返回格式为{"predictions": [{"class_id": ..., "probabilities": [...] }]}
        pred = result['predictions'][0]
        class_id = int(pred['class_id']) if 'class_id' in pred else int(np.argmax(pred['probabilities']))
        probabilities = pred['probabilities']
        predicted_label = id2label.get(class_id, "未知标签")
        confidence = float(probabilities[class_id])
        return jsonify({
            "input_text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_id": class_id
        })
    except Exception as e:
        app.logger.error(f"处理请求时发生未知错误: {e}")
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

# --- 应用启动入口 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 