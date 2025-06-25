import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
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

# --- 加载训练好的模型和分词器 ---
print(f"从 '{MODEL_DIR}' 加载模型和分词器...")
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
print("加载完成。")

# --- 定义推理接口 ---
class ServingModel(tf.Module):
    def __init__(self, model, tokenizer):
        super(ServingModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="texts")])
    def __call__(self, texts):
        """
        推理函数，接收原始文本输入，返回预测结果
        """
        # 1. 使用tf.py_function包装分词，避免在tf.function中追踪python逻辑
        def tokenize_py(texts):
            # texts是一个EagerTensor，需要转为numpy string list
            tokenized_data = self.tokenizer(
                [s.decode('utf-8') for s in texts.numpy()], 
                return_tensors="tf", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            return tokenized_data['input_ids'], tokenized_data['attention_mask'], tokenized_data['token_type_ids']

        # 定义输出类型
        output_signature = (tf.int32, tf.int32, tf.int32)
        
        input_ids, attention_mask, token_type_ids = tf.py_function(
            tokenize_py, 
            [texts], 
            Tout=output_signature
        )

        # 设置输出shape，对于padding=True，第二维是动态的
        input_ids.set_shape([None, None])
        attention_mask.set_shape([None, None])
        token_type_ids.set_shape([None, None])
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        # 2. 模型推理，获取logits
        outputs = self.model(inputs)
        logits = outputs.logits
        
        # 3. 计算概率和预测类别ID
        probabilities = tf.nn.softmax(logits, axis=-1, name='probabilities')
        predicted_class_id = tf.argmax(probabilities, axis=-1, name='predicted_class_id')
        
        # 4. 以字典形式返回结果
        return {
            "probabilities": probabilities,
            "class_id": predicted_class_id
        }

# --- 创建并保存SavedModel ---
print("创建推理模型...")
serving_model = ServingModel(model=model, tokenizer=tokenizer)

print(f"正在将模型导出到 '{EXPORT_PATH}'...")
tf.saved_model.save(
    serving_model,
    EXPORT_PATH,
    signatures=serving_model.__call__.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="texts")
    )
)

print("\n模型已成功导出为SavedModel格式！")
print(f"路径: {EXPORT_PATH}")
print("现在，你可以使用此目录通过TensorFlow Serving启动模型服务。") 