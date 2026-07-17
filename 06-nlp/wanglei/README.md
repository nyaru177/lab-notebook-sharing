---
title: 手写 Transformer 学习笔记
author: 王雷
date: 2026-07-17
tags:
  - nlp
  - transformer
  - pytorch
  - week2
---

# 手写 Transformer 学习笔记

这组笔记配合 [Week 2 手写 Transformer Notebook](week2_handwritten_transformer_wanglei.ipynb) 使用。Notebook 按原始手写练习的学习顺序展开，保留了公式、两张结构示意图和当时遇到问题时的解释；代码使用目录中附带的小型 token 张量，不依赖模型权重或外部数据。

## 辅助资料

1. [基础原理：从张量形状到多头注意力](transformer_foundations.md)
2. [结构组件：从 Embedding 到 Encoder/Decoder](transformer_blocks.md)
3. [实现要点：mask、NaN 与下游任务](transformer_practice_notes.md)
4. 打开 Notebook，按代码顺序对照实现。

## 附带数据

`tensor_src.pt` 和 `tensor_trg.pt` 分别约 38 KB 和 40 KB，内容是形状为 `(128, 36)` 与 `(128, 38)` 的 `torch.int64` source/target token id，不是模型参数。Notebook 使用 `weights_only=True` 和 CPU 读取两者，并取同一 batch 的第一对序列做最小前向示例。

## 适用范围

- 适用周次：Week 2
- 分类：`06-nlp`，相关基础为 `05-pytorch`
- 前置知识：Python、PyTorch 张量、矩阵乘法、Softmax
- 重点：理解 Transformer 的数据流，而不是记忆某个项目的完整工程实现

## 内容来源

本组笔记整理自个人学习记录，并参考：

- [Transformer 全面指南](https://luxiangdong.com/2023/09/10/trans/)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [李沐 Transformer 论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE/)

代码参考B站某视频学习；原始链接找不到了，可以在 B 站搜索“手写 Transformer”学习。

## 视频学习线索

- [【研1基本功（真的很简单）注意力机制】手写多头注意力机制](https://www.bilibili.com/video/BV1o2421A7Dr)
  这是当时学习的“手撕 Transformer 系列”视频入口。
