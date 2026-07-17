---
title: Transformer 实现要点与常见错误
author: 王雷
date: 2026-07-17
tags:
  - transformer
  - debugging
  - pytorch
---

# Transformer 实现要点与常见错误

## 1. 先检查形状，再检查数值

手写 attention 时，最值得先打印或断言的是：

```text
输入                 (batch, sequence, d_model)
拆分多头后的 Q/K/V     (batch, heads, sequence, d_head)
注意力分数            (batch, heads, query_len, key_len)
注意力输出            (batch, heads, query_len, d_head)
拼接后的输出          (batch, query_len, d_model)
```

`d_model` 必须能被 `heads` 整除。`Q` 和 `K` 的最后一维必须相同，`K` 转置后才能与 `Q` 做矩阵乘法；cross-attention 中 `query_len` 和 `key_len` 可以不同。

## 2. 三个容易改变模型行为的错误

### 错误一：把 mask 位置填成正数

错误写法：

```python
scores = scores.masked_fill(mask == 0, 1e9)
```

正数会在 Softmax 后占据极大权重。正确做法是填入当前 dtype 的极小值，并在得到权重后再次清零屏蔽位置：

```python
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
weights = torch.softmax(scores, dim=-1)
weights = weights.masked_fill(~mask, 0.0)
```

### 错误二：把 cross-attention 条件写反

错误逻辑：

```python
if enc_output is None:
    values = cross_attention(values, enc_output, enc_output, cross_mask)
```

`enc_output` 为空时没有可供读取的 Encoder 表示。应当只在它存在时计算 cross-attention：

```python
if encoder_output is not None:
    values = cross_attention(values, encoder_output, encoder_output, cross_mask)
```

### 错误三：默认调用时遗漏 source padding mask

如果调用者没有显式传入 `src_mask`，模型仍应根据源 token 自动构造 padding mask：

```python
if src_mask is None:
    src_mask = make_padding_mask(src_tokens, src_tokens, src_pad_idx, src_pad_idx)
```

否则 source 中的 `<pad>` 会参与 Encoder self-attention，并可能被 Decoder 的 cross-attention 读取。

## 3. 为什么整行 mask 会产生 NaN

如果某个 query 对应的 key 全部被屏蔽，把整行填成 `-inf` 后，Softmax 会遇到 `0 / 0`，结果为 NaN。使用很小的有限值可以避免部分问题，但更根本的约束是：

- 每个 query 至少保留一个可见位置；
- padding mask 与 causal mask 合并后，不能把合法位置全部屏蔽；
- 必要时在 Softmax 前检查 `mask.any(dim=-1)`。

这也是为什么 Notebook 的最小示例只构造合法的 padding/causal mask，并在注意力输出后清零屏蔽权重。

## 4. Q/K/V 在下游任务中的变化

Transformer 的主体通常不变，任务差异主要体现在输入组织方式和输出头：

| 任务 | 注意力形式 | 常见输出 |
| --- | --- | --- |
| 文本分类、情感分析 | 输入序列 self-attention | `[CLS]` 或池化向量 + 分类头 |
| 命名实体识别 | 输入序列 self-attention | 每个 token 一个标签 |
| 翻译、摘要、文本生成 | Decoder masked self-attention + cross-attention | 目标词表 logits |
| 语义匹配、问答 | 两段序列之间的 cross-attention 或联合编码 | 相似度或分类结果 |
| Vision Transformer | 图像 patch 序列的 self-attention | 分类、检测或分割头 |

Q、K、V 的来源决定模型是在序列内部建模，还是在两个序列之间建立对应关系；输出层则决定如何把上下文表示转换为具体任务结果。

## 5. 分享前的检查清单

- 代码单元按顺序执行，不依赖个人路径或隐藏变量。
- 明确 `mask=True` 的语义，并统一 mask 的 dtype 和形状。
- 检查 `d_model`、`n_heads`、序列长度和 padding id。
- 先用小尺寸随机 token 验证输出形状，再接真实数据。
- 说明哪些结果是形状/逻辑断言，哪些才是训练指标。
