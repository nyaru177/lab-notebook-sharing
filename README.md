# 暑期集训 Notebook 共享仓库

这个仓库用于组内共享前三周的编程技巧、学习技术和方向专题 Notebook。目标是让每份内容都做到“别人打开就能读，按顺序运行就能复现”。

## 从哪里开始

| 需求 | 入口 |
|---|---|
| 先看每周示例 | 打开 [`examples/week1/` 到 `examples/week8/`](examples/) 中对应周次的 Notebook |
| 选择一个主题 | 先看下面的分类，再打开对应目录的 `README.md` |
| 新建 Notebook | 复制 [`templates/notebook_template.ipynb`](templates/notebook_template.ipynb) |
| 查看提交步骤 | 阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 提交前自查 | 创建 PR 时使用仓库自动带出的检查清单 |

## 先看示例，再提交

`examples/week1/` 到 `examples/week8/` 是根据《实验室新生暑期居家集训学习计划》整理的每周示例区，每周一个 Notebook。示例刻意只覆盖原文的一小部分，用来展示文件命名、Notebook 开头信息、Markdown 解释、代码组织和结果分析；它们不是完整作业，也不是唯一答案。

实际提交时，先打开对应周次的示例，再把自己的主题、作者、环境、目标、代码和结论替换进去。具体提交位置仍按最匹配的技能分类目录选择。

## 内容分类

下面的分类是检索入口，不是选题边界。每个目录中的主题都只是示例；只要和本目录或集训目标相关，就可以提交。跨分类的内容放到最主要的目录，并在 Notebook 开头注明相关关键词。

| 目录 | 对应内容 | 与《实验室新生暑期居家集训学习计划》的关联 |
|---|---|---|
| `examples/weekX/` | 按 Week 1-8 提供的《实验室新生暑期居家集训学习计划》代码示例和提交样例 | 供组内模仿结构和运行方式，不替代个人交付物 |
| `01-jupyter/` | Jupyter Notebook 使用规范、Markdown、公式、Magic Commands、导出 | Week 1：Jupyter Notebook 系统学习 |
| `02-numpy-scipy/` | NumPy 数组、索引、广播、矩阵运算、向量化练习；SciPy 选学 | Week 1：NumPy 核心，SciPy 常用模块选学 |
| `03-matplotlib/` | 训练曲线、混淆矩阵、条形图、ROC/AUC、t-SNE、图片保存 | Week 1：Matplotlib 可视化 |
| `04-sklearn/` | 数据划分、Pipeline、交叉验证、GridSearchCV、指标、文本分类 | Week 2：Sklearn 机器学习流程 |
| `05-pytorch/` | Tensor、autograd、`nn.Module`、Dataset、DataLoader、训练循环、MNIST | Week 2：PyTorch 深度学习基础 |
| `06-nlp/` | TF-IDF、Word2Vec、Embedding、t-SNE、LSTM/BiLSTM、BERT | NLP 方向学习与实验相关内容 |
| `07-computer-vision/` | OpenCV、PIL、RGB/BGR、边缘检测、CNN、ResNet、视频理解 | CV 方向学习与实验相关内容 |
| `08-git-linux/` | Git/GitHub 工作流、Linux 常用命令、环境操作 | 计划中的编码习惯、协作和提交要求 |
| `optional/` | Docker、LoRA、QLoRA、实验配置、随机种子等扩展内容 | 非主线内容，可作为补充分享 |

## 一次提交的最低要求

- 一个 Notebook 聚焦一个清晰主题，不要求一次覆盖整个分类。
- 开头写清标题、作者、日期、适用周次、关键词、学习目标和运行环境。
- 代码前后有必要的 Markdown 解释，读者能看懂“为什么这样做”和“结果说明什么”。
- 从头运行一次，确认不依赖本地绝对路径、隐藏状态或未提交的文件。
- 外部数据、依赖和参考资料写清来源；数据集、模型权重和缓存不要提交到仓库。

## 和学习计划的关系

《实验室新生暑期居家集训学习计划》要求实验尽量使用 Jupyter Notebook 完成并通过 GitHub 更新进度。这个仓库是小组互相阅读、复用和讨论的公共 Notebook 资源库，不替代个人每周的报告、论文汇报或其他交付物。
