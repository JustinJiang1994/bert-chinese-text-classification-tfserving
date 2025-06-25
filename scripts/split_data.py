import pandas as pd
from sklearn.model_selection import train_test_split

# 读取初始数据
df = pd.read_csv('data/initial.data', sep='\t')

# 先分出训练集（80%）和剩余（20%）
df_train, df_temp = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label'])
# 再将剩余的20%分为验证集和测试集（各10%）
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp['label'])

# 保存
for subset, name in zip([df_train, df_val, df_test], ['train', 'val', 'test']):
    subset.to_csv(f'data/{name}.data', sep='\t', index=False)
    print(f'{name}.data 保存完成，样本数：{len(subset)}') 