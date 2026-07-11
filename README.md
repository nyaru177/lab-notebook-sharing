# 暑期集训 Notebook 共享仓库

这个仓库用于组内共享前三周的编程技巧、学习技术和方向专题 Notebook。每位同学提交的内容应尽量做到“别人打开就能读，按顺序运行就能复现”。

## 内容分类

| 目录 | 对应内容 | 来自《实验室新生暑期居家集训学习计划》的要求 |
|---|---|---|
| `01-jupyter/` | Jupyter Notebook 使用规范、Markdown、公式、Magic Commands、导出 | Week 1：Jupyter Notebook 系统学习 |
| `02-numpy-scipy/` | NumPy 数组、索引、广播、矩阵运算、向量化练习；SciPy 选学 | Week 1：NumPy 核心，SciPy 常用模块选学 |
| `03-matplotlib/` | 训练曲线、混淆矩阵、条形图、ROC/AUC、t-SNE、图片保存 | Week 1：Matplotlib 可视化 |
| `04-sklearn/` | 数据划分、Pipeline、交叉验证、GridSearchCV、指标、20 Newsgroups | Week 2：Sklearn 机器学习流程 |
| `05-pytorch/` | Tensor、autograd、`nn.Module`、Dataset、DataLoader、训练循环、MNIST | Week 2：PyTorch 深度学习基础 |
| `06-nlp/` | TF-IDF、Word2Vec、Embedding、t-SNE、LSTM/BiLSTM、BERT | Week 3-5：NLP 方向学习与实验 |
| `07-computer-vision/` | OpenCV、PIL、RGB/BGR、边缘检测、CNN、ResNet、视频理解 | Week 3-5：CV 方向学习与实验 |
| `08-git-linux/` | Git/GitHub 工作流、Linux 常用命令、环境操作 | 原始计划的编码习惯和仓库提交要求 |
| `optional/` | Docker、LoRA、QLoRA、实验配置、随机种子等扩展内容 | 非原始主线，可作为补充分享 |

## 提交规范

1. 每个 Notebook 放到最匹配的分类目录中。
2. 文件名建议使用：`weekX_主题_姓名.ipynb`，例如 `week2_pipeline_zhangsan.ipynb`。
3. Notebook 顶部必须写清楚标题、作者、日期、适合人群、学习目标和运行环境。
4. 所有核心代码前后都要有 Markdown 说明，不要只堆代码。
5. 提交前至少从头运行一次，确认没有明显报错。
6. 大型数据集、模型权重、缓存文件不要直接提交到仓库。

## 推荐 Notebook 结构

```text
1. 标题、作者、日期、学习目标
2. 背景说明：这个技巧解决什么问题
3. 环境与依赖：Python 版本、主要库版本
4. 最小可运行示例
5. 关键概念解释
6. 练习或变体实验
7. 常见错误与排查
8. 总结与参考资料
```

## 和《实验室新生暑期居家集训学习计划》的关系

《实验室新生暑期居家集训学习计划》要求所有实验尽量使用 Jupyter Notebook 完成并通过 GitHub 更新进度。这个仓库不是替代个人每周交付物，而是用于小组互相阅读、复用和讨论的公共 Notebook 资源库。
