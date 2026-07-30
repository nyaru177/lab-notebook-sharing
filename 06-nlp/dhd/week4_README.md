# NLP 模型演进学习项目：RNN → Transformer → BERT

本项目是 Week 4 的学习实践，旨在通过三个 Jupyter Notebook，依次实现 BiLSTM、Transformer 和 BERT 文本分类器，帮助深入理解 NLP 模型的演进脉络。

## 文件结构

- `week4_Workflow.ipynb`：顶层导航，介绍整体工作流和环境。
- `week4_BiLSTM-Classifier.ipynb`：手写 BiLSTM + 注意力池化，在 AG_NEWS 上训练。
- `week4_Transformer-FromScratch.ipynb`：从零实现 Transformer 编码器，同样在 AG_NEWS 上训练。
- `week4_BERT-Finetune.ipynb`：使用 HuggingFace 微调 BERT，在 IMDB 上训练。
- `week4_nlp_helpers.py`：辅助函数（绘图、评估等）。
- `intermediate_data/`：存放模型权重、分词器等中间产物（运行后生成）。

## 快速开始

1. 创建虚拟环境（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows