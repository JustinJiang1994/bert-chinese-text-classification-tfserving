import requests
import json
import numpy as np
import os
from transformers import BertTokenizer

# --- 配置 ---
URL = "http://localhost:8501/v1/models/bert-chinese:predict"
TOKENIZER_DIR = 'saved_model/bert-chinese'
LABEL_MAP_PATH = os.path.join(TOKENIZER_DIR, 'label2id.txt')

# --- 加载分词器和标签映射 ---
if not os.path.exists(TOKENIZER_DIR):
    print(f"错误: 分词器目录 '{TOKENIZER_DIR}' 不存在。")
    print("请先确保已成功运行 train_bert.py。")
    exit()

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
def load_label_map(path):
    if not os.path.exists(path):
        print(f"错误: 标签映射文件 '{path}' 不存在。")
        print("请先确保已成功运行 train_bert.py。")
        exit()
        
    id2label = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label, idx = line.strip().split('\t')
            id2label[int(idx)] = label
    return id2label
id2label = load_label_map(LABEL_MAP_PATH)


# --- 测试数据 ---
test_texts = [
    "这手机拍照真好看，我很喜欢！",
    "电池太不耐用了，一天要充好几次电。",
    "手机屏幕显示效果还行，中规中矩。"
]

# --- 客户端分词 ---
print("正在客户端进行分词...")
tokenized_inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="np" # 使用numpy数组
)

# --- 构建请求体 ---
# 将numpy数组转为list，因为JSON不支持numpy类型
instances = [{
    "input_ids": tokenized_inputs['input_ids'][i].tolist(),
    "attention_mask": tokenized_inputs['attention_mask'][i].tolist(),
    "token_type_ids": tokenized_inputs['token_type_ids'][i].tolist(),
} for i in range(len(test_texts))]

request_body = {
    "signature_name": "serving_default",
    "instances": instances
}

# --- 发送请求 ---
print(f"向 {URL} 发送请求...")
try:
    response = requests.post(URL, data=json.dumps(request_body))
    response.raise_for_status()

    # --- 解析并打印结果 ---
    predictions = response.json().get('predictions')
    if predictions is None:
        raise KeyError
    
    print("\n模型预测结果:\n" + "="*30)
    for i, text in enumerate(test_texts):
        class_id = predictions[i]['class_id']
        probabilities = predictions[i]['probabilities']
        
        predicted_label = id2label.get(class_id, "未知标签")
        confidence = probabilities[class_id]
        
        print(f"文本: '{text}'")
        print(f"  -> 预测类别: {predicted_label} (置信度: {confidence:.4f})")
        print("-" * 30)

except requests.exceptions.RequestException as e:
    print(f"\n请求失败: {e}")
    print("请确认 TensorFlow Serving 容器正在运行，并且端口映射正确。")
    print("详细错误:", response.text if 'response' in locals() else "无响应内容")
except (KeyError, IndexError, TypeError):
    print(f"\n返回结果格式错误或解析失败，收到的内容: {response.text}") 