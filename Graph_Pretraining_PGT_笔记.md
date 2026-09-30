# Graph Pre-training 与 PGT 学习笔记

> 论文：**Generalizing Graph Transformers Across Diverse Graphs and Tasks via Pre-Training on Industrial-Scale Data**
>
> 方法：**PGT（Pre-trained Graph Transformer）**

---

## 1. 为什么需要 Graph Pre-training？

传统 GNN 通常是：

$$
\text{一个图}\rightarrow\text{一个模型}\rightarrow\text{一个任务}
$$

例如：

$$
\text{Cora}\rightarrow\text{GAT}\rightarrow\text{节点分类}
$$

Graph Pre-training 希望先利用大量图数据进行**自监督预训练**，学习比较通用的图表示能力，再迁移到不同的下游任务：

$$
\boxed{
\text{大量图数据}
\rightarrow
\text{自监督预训练}
\rightarrow
\text{Graph Encoder}
\rightarrow
\text{下游任务}
}
$$

核心不是简单“加工图”，而是**训练一个能够理解图结构和节点特征的 Encoder**。

---

## 2. Encoder、Decoder 和 Transformer

### Encoder

Encoder 的作用：

> 把原始输入转换成更有用的 Representation。

$$
(X,G)\xrightarrow{\text{Encoder}}H
$$

在 GNN 中，GCN、GAT 都可以充当 Encoder。

### Decoder

Decoder 根据 Encoder 得到的表示完成恢复或预测：

$$
X\rightarrow Encoder\rightarrow H\rightarrow Decoder\rightarrow\hat X
$$

### Transformer

Transformer 的核心是 **Self-Attention**：

$$
h_i=\sum_j\alpha_{ij}V_j
$$

即根据不同节点的重要程度进行信息聚合。

---

## 3. GAT、Transformer 和 Graph Transformer

### GAT

GAT 的 Attention 受到图结构限制：

$$
h_i'=\sigma\left(\sum_{j\in N_i}\alpha_{ij}Wh_j\right)
$$

只有邻居节点才能参与聚合。

### Transformer

普通 Transformer 中，一个元素通常可以和其他元素建立 Attention。

### Graph Transformer

可以理解为：

$$
\boxed{\text{Graph Structure}+\text{Transformer Attention}}
$$

即利用 Transformer 的 Attention 能力来建模图中的节点关系。

---

# 4. PGT 的核心思想

PGT 针对的是**超大规模图**，不能直接把整个图送进 Transformer。

整体流程：

$$
\boxed{
\text{Large Graph}
\rightarrow
\text{PPR}
\rightarrow
\text{Node Sequence}
\rightarrow
\text{Masking}
\rightarrow
\text{Graph Transformer}
\rightarrow
\text{Pre-training}
}
$$

### 4.1 PPR：寻找重要上下文节点

给定 Seed Node：

$$
v_s
$$

通过 Personalized PageRank（PPR）寻找与它最相关的 Top-k 节点：

$$
C_s=\text{Top-k nodes according to PPR}
$$

然后构造：

$$
S_s=(v_s)\cup C_s
$$

可以理解为：

> **以一个节点为中心，选出最重要的一批上下文节点，形成一个局部图结构。**

### 4.2 Masking

随机遮住一部分节点：

$$
[v_1,v_2,v_3,v_4]
\rightarrow
[v_1,\boxed{[M]},v_3,\boxed{[M]}]
$$

然后把未被 Mask 的节点送入 Graph Transformer Encoder。

### 4.3 两个预训练任务

**① Feature Reconstruction**

让模型根据上下文恢复被 Mask 的节点特征：

$$
H\xrightarrow{Decoder}\hat X
$$

使：

$$
\hat X\approx X
$$

得到损失：

$$
L_1
$$

**② Local Structure Reconstruction / Contrastive Learning**

同一个局部 sequence 中的节点相互接近，不同 sequence 中的节点相互区分：

$$
sim(h_i,h_j)\uparrow\quad\text{(positive)}
$$

$$
sim(h_i,h_j)\downarrow\quad\text{(negative)}
$$

得到：

$$
L_2
$$

最终：

$$
\boxed{L=L_1+\lambda L_2}
$$

---

# 5. PGT 训练完成后

预训练完成后得到：

$$
\boxed{\text{Pre-trained Graph Encoder}}
$$

然后迁移到下游任务：

$$
\text{Pre-trained Encoder}
\rightarrow
\begin{cases}
\text{节点分类}\\
\text{推荐}\\
\text{社区检测}\\
\text{其他任务}
\end{cases}
$$

因此：

> **Graph Pre-training 的核心就是先学“如何表示图”，再把这种能力迁移给具体任务。**

---

# 6. DeepWalk → GCN → GAT → Graph Transformer → Graph Pre-training

```text
DeepWalk
  ↓
Random Walk + Skip-Gram
  ↓
学习节点表示

GCN
  ↓
邻居聚合

GraphSAGE
  ↓
更灵活、可采样的邻居聚合

GAT
  ↓
Attention 加权邻居

Graph Transformer
  ↓
Transformer Attention + Graph Structure

Graph Pre-training
  ↓
先学习通用 Graph Encoder

PGT
  ↓
PPR + Graph Transformer
+ Masked Reconstruction
+ Contrastive Learning
  ↓
Pre-trained Graph Encoder
  ↓
不同图 / 不同下游任务
```

---

# 7. 一句话理解 PGT

> **PGT 先利用 PPR 从超大图中找到与节点最相关的上下文，再用 Graph Transformer 对局部结构进行编码，并通过 Masked Reconstruction 和 Contrastive Learning 进行自监督预训练，从而得到可以迁移到不同图和下游任务的 Graph Encoder。**

最核心的主线：

$$
\boxed{
\text{PPR}
\rightarrow
\text{Local Structure}
\rightarrow
\text{Graph Transformer}
\rightarrow
\text{Self-supervised Pre-training}
\rightarrow
\text{Transfer}
}
$$
