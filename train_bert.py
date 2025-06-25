import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf_keras as keras

# 1. 读取数据
def load_data(path):
    return pd.read_csv(path, sep='\t')

train_df = load_data('data/train.data')
val_df = load_data('data/val.data')
test_df = load_data('data/test.data')

# 2. 标签映射
all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique()
label2id = {l: i for i, l in enumerate(sorted(all_labels))}
id2label = {i: l for l, i in label2id.items()}
train_df['label'] = train_df['label'].map(label2id)
val_df['label'] = val_df['label'].map(label2id)
test_df['label'] = test_df['label'].map(label2id)

# 3. 加载分词器
MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 4. 编码数据
def encode(df):
    return tokenizer(
        df['txt'].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )

train_encodings = encode(train_df)
val_encodings = encode(val_df)
test_encodings = encode(test_df)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_df['label'].values
)).shuffle(len(train_df)).batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_df['label'].values
)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_df['label'].values
)).batch(32)

# 5. 加载模型
model = TFBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id)
)

# 6. 训练
optimizer = keras.optimizers.Adam(learning_rate=2e-5)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# 7. 保存模型
os.makedirs('saved_model/bert-chinese', exist_ok=True)
model.save_pretrained('saved_model/bert-chinese')
tokenizer.save_pretrained('saved_model/bert-chinese')

# 8. 测试集评估
test_results = model.evaluate(test_dataset)
print('Test loss, Test accuracy:', test_results)

# 9. 输出标签映射，便于后续推理
with open('saved_model/bert-chinese/label2id.txt', 'w', encoding='utf-8') as f:
    for label, idx in label2id.items():
        f.write(f'{label}\t{idx}\n') 