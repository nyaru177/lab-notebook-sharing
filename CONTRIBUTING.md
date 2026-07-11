# 贡献说明

## 提交流程

1. 先同步最新仓库内容。
2. 在对应分类目录中新建或更新自己的 Notebook。
3. 按顺序从头运行 Notebook。
4. 确认输出、图片和说明完整。
5. 提交 commit，并在提交信息里写清楚主题。

示例：

```bash
git pull
git add 03-matplotlib/week1_tsne_yourname.ipynb
git commit -m "Add t-SNE visualization notebook"
git push
```

## Notebook 内容要求

- 开头必须包含：标题、作者、日期、适用周次、学习目标、运行环境。
- 代码应能按顺序执行，不依赖隐藏状态。
- 关键图表要有标题、坐标轴、图例或文字说明。
- 如果使用外部数据，说明下载来源和放置路径。
- 如果代码来自教程或官方文档，需要在最后列出参考链接。

## 不建议提交的内容

- `.ipynb_checkpoints/`
- `__pycache__/`
- `.venv/`、`env/`
- 大型数据集、模型权重、缓存文件
- 含账号、Token、Cookie 或个人隐私的信息

## 评阅读者关注点

别人阅读你的 Notebook 时，应能快速回答：

- 这个 Notebook 教会我什么？
- 核心概念和代码是否对应？
- 我能不能复制一份后直接运行？
- 是否和《实验室新生暑期居家集训学习计划》中的技能点有关？
