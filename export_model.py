import tensorflow as tf
from transformers import TFBertForSequenceClassification
import os
import shutil

# --- 配置 ---
MODEL_DIR = 'saved_model/bert-chinese'
SERVING_MODEL_DIR = 'tf_serving_model/bert-chinese'
MODEL_VERSION = '1'
EXPORT_PATH = os.path.join(SERVING_MODEL_DIR, MODEL_VERSION)

# --- 清理旧模型 ---
if os.path.exists(SERVING_MODEL_DIR):
    shutil.rmtree(SERVING_MODEL_DIR)

# --- 加载训练好的模型 ---
print(f"从 '{MODEL_DIR}' 加载模型...")
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
print("加载完成。")

# --- 定义只接受张量输入的推理接口 ---
@tf.function
def serving_fn(input_ids, attention_mask, token_type_ids):
    """
    纯模型的推理函数，只接受张量输入。
    """
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }
    outputs = model(inputs, training=False)
    logits = outputs.logits
    
    probabilities = tf.nn.softmax(logits, axis=-1, name='probabilities')
    predicted_class_id = tf.argmax(probabilities, axis=-1, name='predicted_class_id')
    
    return {
        "probabilities": probabilities,
        "class_id": predicted_class_id
    }

# --- 获取函数签名 ---
concrete_function = serving_fn.get_concrete_function(
    input_ids=tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
    attention_mask=tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask"),
    token_type_ids=tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="token_type_ids")
)


# --- 创建并保存SavedModel ---
print(f"正在将模型导出到 '{EXPORT_PATH}'...")
tf.saved_model.save(
    model,
    EXPORT_PATH,
    signatures={'serving_default': concrete_function}
)

print("\n模型已成功导出为SavedModel格式！")
print(f"路径: {EXPORT_PATH}")
print("此模型只接受张量输入，分词步骤需在客户端完成。") 