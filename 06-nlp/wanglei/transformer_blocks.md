---
title: Transformer 结构组件
author: 王雷
date: 2026-07-17
tags:
  - transformer
  - encoder-decoder
  - pytorch
---

# Transformer 结构组件

## 1. Token Embedding 与位置编码

Embedding 将离散 token id 映射为连续向量：

```python
token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
```

注意力本身对 token 的排列是等变的：如果只交换输入顺序，模型没有办法凭注意力知道原来的位置。因此需要加入位置编码。经典 Transformer 使用正弦、余弦编码：

$$
PE_{pos,2i}=\sin\left(pos/10000^{2i/d_{model}}\right)
$$
$$
PE_{pos,2i+1}=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

实现时通常先生成 `(max_len, d_model)` 的位置表，再切出当前序列长度，与 token embedding 相加。这样每个位置同时包含词义信息和顺序信息。

## 2. Position-wise Feed-Forward Network

每个 Encoder/Decoder 层都包含一个对每个位置独立应用的前馈网络：

$$
\operatorname{FFN}(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2
$$

代码通常是两个线性层和一个激活函数：

```python
nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(d_ff, d_model),
)
```

注意输入输出维度回到 `d_model`，才能与残差连接相加；`d_ff` 提供更宽的中间表示，ReLU 引入非线性，Dropout 用于正则化。

## 3. LayerNorm、残差与 Dropout

对最后一个特征维做 LayerNorm。给定一个位置的向量 `x`：

$$
\operatorname{LayerNorm}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

`gamma` 和 `beta` 是可学习参数。与 BatchNorm 不同，LayerNorm 不依赖 batch 维度的统计量，因此训练和推理时都可以对每个样本、每个位置独立归一化。

Transformer 子层通常通过残差连接和归一化保持稳定：

```python
x = norm(x + dropout(sublayer(x)))
```

残差连接保留原始信号并改善梯度传播；LayerNorm 控制激活尺度；Dropout 降低过拟合风险。实际代码要注意：使用方差计算时应采用总体方差语义（例如 `unbiased=False`），并为分母加入 `eps`。

## 4. Encoder Layer

一个 Encoder layer 的数据流是：

```text
x
 ├─ Multi-Head Self-Attention(src_padding_mask)
 ├─ Add & Norm
 ├─ Position-wise FFN
 └─ Add & Norm
```

多个 Encoder layer 逐层处理源序列，输出每个源位置的上下文表示。Encoder 只做 self-attention，不需要 cross-attention。

## 5. Decoder Layer

一个 Decoder layer 多一个与 Encoder 输出交互的子层：

```text
y
 ├─ Masked Multi-Head Self-Attention(target_mask)
 ├─ Add & Norm
 ├─ Cross-Attention(Q=decoder, K/V=encoder, source_padding_mask)
 ├─ Add & Norm
 ├─ Position-wise FFN
 └─ Add & Norm
```

Cross-attention 的 Query 来自 Decoder 当前状态，Key 和 Value 来自 Encoder 输出。它让生成过程能够回看源序列中与当前目标位置相关的信息。最后通过线性层映射到目标词表，得到每个位置的 logits。

经典 Transformer 是 Encoder-Decoder 架构，所以 Decoder 确实包含 cross-attention；只有 decoder-only 的语言模型才会省略 Encoder 和 cross-attention。
