# 中文文本分类（BERT + TensorFlow Serving）

## 项目简介
本项目旨在基于BERT模型进行中文文本分类，最终通过TensorFlow Serving部署模型服务。当前已实现数据集的切分，后续将补充模型训练与服务部署部分。

## 数据说明
- 数据文件位于`data/`目录。
- 原始数据文件为`initial.data`，格式如下（以Tab分隔）：

```
label\ttxt
1\t手机很不错，玩游戏很流畅，快递小哥态度特别特别特别好，重要的事情说三遍?
0\t电池一直用可以用半天，屏幕很好。
-1\t一个月都有点卡了，
```
- `label`为类别标签（如1/0/-1），`txt`为中文评论文本。

## 数据集切分
- 使用`scripts/split_data.py`脚本将`initial.data`按8:1:1比例切分为训练集（train.data）、验证集（val.data）、测试集（test.data）。
- 切分后数据仍为Tab分隔，格式与原始数据一致。

### 运行方法
1. 安装依赖：
   ```bash
   pip install pandas==2.1.1 scikit-learn==1.6.1
   ```
2. 执行切分脚本：
   ```bash
   python scripts/split_data.py
   ```
3. 运行后将在`data/`目录下生成`train.data`、`val.data`、`test.data`。

## 后续开发计划
- 基于BERT的文本分类模型训练（TensorFlow/Keras）。
- 模型导出为SavedModel格式。
- 使用TensorFlow Serving部署模型，提供在线推理API。
- 增加推理/评测脚本与接口文档。

## 目录结构
```
bert-chinese-text-classification-tfserving/
  ├── data/
  │   ├── initial.data
  │   ├── train.data
  │   ├── val.data
  │   └── test.data
  └── scripts/
      └── split_data.py
```
