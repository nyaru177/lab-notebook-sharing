---
title: Transformer 基础原理
author: 王雷
date: 2026-07-17
tags:
  - transformer
  - attention
  - tensor-shape
---

# Transformer 基础原理

## 1. 先统一张量形状

本笔记和 Notebook 统一使用 `batch_first=True`，序列张量形状为：

```text
(batch, sequence, feature)
```

- `batch`：一次并行处理的样本数。
- `sequence`：token 或时间步的数量。
- `feature`：每个 token 的表示维度，通常就是 `d_model`。

例如 `(128, 64, 512)` 表示 128 个样本、每个样本 64 个 token、每个 token 使用 512 维表示。多头注意力会把最后一维拆成 `n_heads` 个头，因此必须满足：

```text
d_model % n_heads == 0
d_head = d_model / n_heads
```

## 2. Q、K、V：注意力在比较什么

给定输入表示 `X`，三个线性层产生：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

- **Query**：当前位置想查询什么信息。
- **Key**：每个位置可以被匹配的特征。
- **Value**：匹配之后真正汇聚的内容。

缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

`QK^T` 的每个元素表示一个 query 与一个 key 的相似度。除以 `sqrt(d_k)` 是为了控制点积的尺度，避免维度变大后 Softmax 过早饱和。

如果 Q、K、V 来自同一序列，这是 **self-attention**；如果 Q 来自解码器、K/V 来自编码器输出，这是 **cross-attention**。

## 3. 为什么 `matmul` 只看最后两维

对于高维张量，`torch.matmul` 把最后两维当作矩阵维度，前面的维度作为批次维度进行广播。例如：

```python
q.shape == (batch, heads, query_len, d_head)
k.shape == (batch, heads, key_len, d_head)
scores = q @ k.transpose(-2, -1)
scores.shape == (batch, heads, query_len, key_len)
```

因此每个 batch、每个 head 都独立完成一次矩阵乘法；`batch` 和 `heads` 不参与最后两维的乘法，只负责索引并行计算的矩阵。

## 4. 多头注意力

多头注意力不是把一个注意力层重复堆叠，而是把表示拆成多个子空间并行计算：

1. 分别对输入做 `W_Q`、`W_K`、`W_V` 线性投影。
2. 将 `(batch, sequence, d_model)` 变形为 `(batch, heads, sequence, d_head)`。
3. 每个 head 独立计算缩放点积注意力。
4. 拼接所有 head，再通过输出投影 `W_O` 混合信息。

多个 head 可以从不同表示子空间和不同位置关系中提取信息。由于每个 head 的维度变小，总体计算量仍与 `d_model` 同量级。

## 5. Mask 的两种语义

本项目约定：`True` 表示允许注意，`False` 表示屏蔽。

### Padding mask

不同长度的序列通常用 `<pad>` 补齐。Padding mask 防止模型把注意力分配给这些无意义位置。对于 encoder self-attention，形状通常可广播为：

```text
(batch, 1, 1, source_len)
```

### Causal mask

Decoder 生成第 `t` 个 token 时不能看到未来 token。下三角矩阵可以保留当前位置和过去的位置：

```python
causal_mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
```

Decoder 的 self-attention 通常使用：

```python
target_mask = target_padding_mask & causal_mask
```

### Mask 的填充

将不可见分数填成很小的数，再做 Softmax：

```python
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

注意力权重计算后还可以再次将不可见位置清零，避免数值误差带来残余权重。不要把不可见位置填成正的 `1e9`；这样会让 Softmax 反而偏向被屏蔽的位置。
