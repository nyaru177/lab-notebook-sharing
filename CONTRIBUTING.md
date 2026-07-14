# 贡献说明

## 推荐提交流程

默认使用“个人分支 + Pull Request”，这样每个人可以独立提交，其他同学也能在合并前阅读和讨论。

1. 同步最新的 `main` 分支。
2. 先在各分类目录中打开对应周次的 `*_example.ipynb`，照着它的首页信息、Markdown 说明和代码组织来写；需要空白起点时再复制 `templates/notebook_template.ipynb`。
3. 选择最匹配的分类目录，按命名规则保存 Notebook。
4. 从头运行 Notebook，保存必要的输出并删除无关的大型输出。
5. 创建个人分支，提交 Notebook 并推送到 GitHub。
6. 创建 Pull Request，填写检查清单，等待至少一位同学快速阅读。

如果当前小组约定直接提交 `main`，也应保留下面的命名、内容和运行检查要求。

## 文件命名与分类

建议使用英文小写、下划线分隔的文件名：

```text
weekX_topic_name.ipynb
```

例如：

```text
04-sklearn/week2_pipeline_zhangsan.ipynb
03-matplotlib/week1_tsne_lisi.ipynb
```

主题只需要聚焦一个明确问题。目录中的“可分享主题”是示例，不是限制；如果主题跨多个方向，放入最主要的目录，并在开头写出相关分类和关键词。

## Notebook 最低内容

- 标题、作者、日期、适用周次、分类、关键词。
- 学习目标和必要的前置知识。
- 背景说明：这个方法解决什么问题。
- Python 版本、主要依赖、数据来源和运行方式。
- 最小可运行示例，以及对关键代码的解释。
- 实验结果、图表说明、常见错误或注意事项。
- 总结和参考资料链接。

代码应能按顺序执行，不依赖隐藏状态、个人电脑上的绝对路径或未提交的本地文件。图表应有标题、坐标轴、图例或文字说明。

## 提交前检查

- [ ] 文件放在最匹配的分类目录中，且文件名包含周次、主题和姓名。
- [ ] Notebook 开头信息完整，目标和运行环境清楚。
- [ ] 已执行“Restart Kernel and Run All”，没有报错。
- [ ] 外部数据、依赖和参考链接已说明。
- [ ] 没有账号、Token、Cookie、个人隐私或本地绝对路径。
- [ ] 没有提交 `.ipynb_checkpoints`、虚拟环境、数据集、模型权重和缓存文件。
- [ ] 输出结果大小适中，读者可以直接阅读。

## Git 提交示例

```bash
git switch main
git pull
git switch -c add-zhangsan-sklearn-pipeline
git add 04-sklearn/week2_pipeline_zhangsan.ipynb
git commit -m "Add sklearn pipeline notebook"
git push -u origin add-zhangsan-sklearn-pipeline
```

提交信息建议使用 `Add <topic> notebook` 或 `Update <topic> notebook`，一次 Pull Request 尽量只包含一个主题或一组紧密相关的 Notebook。提交前应先参考对应周次的 `_example.ipynb`，模板只是结构兜底。

## 不建议提交的内容

- `.ipynb_checkpoints/`
- `__pycache__/`
- `.venv/`、`venv/`、`env/`
- 大型数据集、模型权重、缓存文件和自动生成的临时目录
- 含账号、Token、Cookie 或个人隐私的信息
