# 今日学习笔记：Graph Prompt Learning 与 Graph Sparse Prompting

> 论文：**Reliable and Compact Graph Fine-tuning via Graph Sparse Prompting**  
> Authors: Bo Jiang, Hao Wu, Beibei Wang, Jin Tang, Bin Luo  
> arXiv: 2410.21749v1

---

## 1. 今天最重要的主线

这篇论文真正要解决的问题可以压缩成一条线：

```text
预训练 GNN
   ↓
冻结预训练模型参数 Φ
   ↓
下游任务与预训练任务存在差异
   ↓
用 Prompt 改造输入图
   ↓
让原来的 GNN 更适合下游任务
   ↓
发现“所有节点特征维度都加 Prompt”可能很浪费
   ↓
只选择真正有用的特征维度
   ↓
Graph Sparse Prompting (GSP)
   ├── GSFP：一个共享 Prompt + 稀疏特征选择
   └── GSmFP：多个 Prompt + 行稀疏
```

**一句话：**

> Graph Prompt Learning 不主要改变“模型怎么处理图”，而是通过改变“输入给模型的图”，让已经预训练好的 GNN 更好地适应新的任务。

而本文进一步问：

> 既然要修改输入图，那么是不是没必要把所有节点特征维度都修改？能不能只修改真正有用的部分？

---

# 2. 首先要分清三个东西：图、模型、模型参数

## 2.1 图是什么？

论文中的图可以写成：

$$
G=(X,A)
$$

其中：

- $X$：节点特征矩阵
- $A$：邻接矩阵，表示节点之间的连接关系

例如：

$$
X=
\begin{bmatrix}
 x_{11}&x_{12}&x_{13}\\
 x_{21}&x_{22}&x_{23}\\
 x_{31}&x_{32}&x_{33}
\end{bmatrix}
$$

3 个节点，每个节点有 3 个特征。

---

## 2.2 GNN 模型是什么？

可以抽象成：

$$
y=f_\Phi(G)
$$

这里：

- $G$：输入图
- $f_\Phi$：GNN
- $\Phi$：GNN 的可学习参数
- $y$：模型输出

例如 GraphSAGE 中，模型会学习：

```text
自己的节点特征
       ↓
邻居节点特征
       ↓
聚合邻居信息
       ↓
更新节点表示
       ↓
分类 / 预测
```

所以：

> **模型参数 Φ 决定“怎么处理图”。**

而：

> **图 G 决定“处理什么数据”。**

---

# 3. 什么叫“模型能力”？——用 GraphSAGE 理解

假设有一个社交网络：

- 节点 = 用户
- 边 = 好友关系
- 节点特征 = 年龄、发帖数量、游戏时间、科技内容占比……
- 任务 = 判断一个用户是不是科技兴趣用户

GraphSAGE 学到的并不是：

> “用户 A 是科技用户，用户 B 不是科技用户。”

而更像是：

> “如果一个人的自身特征怎样，并且他的邻居具有怎样的特征，那么这个人可能属于什么类别。”

也就是一种输入 → 输出的规则：

$$
\text{自身特征}+\text{邻居信息}
\longrightarrow
\text{节点表示}
\longrightarrow
\text{预测结果}
$$

训练完成后，参数 $\Phi^*$ 中就编码了这种处理规则。

因此所谓“模型能力”，可以简单理解成：

> **参数 $\Phi$ 所代表的从输入图到输出结果的处理规律。**

---

# 4. Fine-tuning 与 Prompt Learning 的区别

这是理解整篇论文最重要的地方之一。

## 4.1 普通 Fine-tuning

预训练得到：

$$
\Phi_{pre}
$$

下游任务继续训练：

$$
\Phi_{pre}\rightarrow\Phi_{down}
$$

也就是说：

> **直接修改 GNN 本身的参数。**

例如：

```text
预训练 GNN
   ↓
修改 W、Attention、MLP 等参数
   ↓
得到适应下游任务的新 GNN
```

优点：灵活。  
缺点：需要更新大量参数，计算和存储成本更高，而且预训练知识可能发生不理想的改变。

---

## 4.2 Graph Prompt Learning

Prompt Learning 的思路是：

$$
\Phi_{pre}\quad\text{保持不变}
$$

然后改变输入：

$$
G\rightarrow\tilde G
$$

于是：

$$
\boxed{
G\rightarrow\tilde G\rightarrow f_{\Phi_{pre}}(\tilde G)
}
$$

也就是说：

> **不大规模修改预训练 GNN，而是学习一个 Prompt，让输入图变得更适合这个已经学会的模型。**

---

# 5. “改变节点特征，不就是改变图了吗？”——是的

这是今天讨论中特别重要的一个问题。

如果：

$$
G=(X,A)
$$

Prompt 修改节点特征：

$$
X\rightarrow\tilde X
$$

那么新的输入确实是：

$$
\tilde G=(\tilde X,A)
$$

所以：

> **Prompt Learning 确实改变了输入图。**

但它没有改变预训练 GNN 的参数：

$$
\Phi_{pre}\quad\text{不变}
$$

因此要区分：

| 东西 | Prompt Learning 中是否改变 |
|---|---|
| 输入节点特征 $X$ | 可以改变 |
| 输入图 $G$ | 可以改变 |
| 邻接关系 $A$ | 视 Prompt 方法而定 |
| 预训练 GNN 参数 $\Phi$ | 通常冻结 |

所以“改变图”和“改变模型”不是一回事。

---

# 6. 为什么要 Prompt？

论文指出，预训练和下游任务之间可能存在 **domain/task discrepancy（领域/任务差异）**。

例如：

```text
预训练阶段
    ↓
模型学到某种通用图表示能力
    ↓
下游任务
    ↓
数据分布 / 任务目标发生变化
    ↓
直接使用原来的模型可能不够合适
```

Prompt Learning 的目标就是：

> **通过改变输入图或任务形式，缩小预训练任务与下游任务之间的差距。**

---

# 7. Graph Prompt 可以修改什么？

论文介绍的 Graph Prompt 方法并不只有一种。

主要可以分成两类：

## 7.1 Task Aligning Prompting

不一定修改图本身，而是修改任务形式，让下游任务和预训练任务更一致。

例如论文提到：

- GPPT
- GraphPrompt

核心思想：

> **重新组织任务，使预训练模型原来擅长的任务形式能够迁移到新的任务。**

---

## 7.2 Graph Data Prompting

直接修改图数据。

例如：

- 调整边权
- 添加 Prompt 节点
- 添加 Prompt 边
- 修改节点特征
- 修改隐藏层表示

本文主要研究的是：

> **节点特征 Prompting。**

---

# 8. 本文重点：GPF

本文把 GPF 作为重点基础方法之一。

设原始节点特征：

$$
X\in\mathbb R^{n\times d}
$$

学习一个 Prompt 向量：

$$
 p\in\mathbb R^d
$$

然后给所有节点都加上同一个 Prompt：

$$
\boxed{
\tilde X=X+\mathbf1p^T
}
$$

其中：

- $n$：节点数量
- $d$：节点特征维度
- $p$：可学习 Prompt
- $\mathbf1$：长度为 $n$ 的全 1 向量

---

# 9. GPF 到底做了什么？

假设一个节点原来是：

$$
 x_i=[2,5,1,8]
$$

Prompt：

$$
 p=[0.1,-0.2,0,0.3]
$$

那么：

$$
\tilde x_i=x_i+p
$$

得到：

$$
\tilde x_i=[2.1,4.8,1,8.3]
$$

所有节点都做类似的操作。

因此 GPF 的思想非常直观：

> **在不修改预训练 GNN 参数的情况下，给输入节点特征加入一个可学习的偏移量。**

---

# 10. GPF 为什么可能有问题？

关键问题：

> **是不是所有特征维度都值得修改？**

例如一个节点有 100 个特征：

$$
[x_1,x_2,\cdots,x_{100}]
$$

GPF 会学习：

$$
[p_1,p_2,\cdots,p_{100}]
$$

然后全部加进去。

但实际上可能只有：

$$
 p_3,p_{17},p_{42}
$$

真正有用。

其余维度：

$$
 p_1=p_2=p_4=\cdots=0
$$

可能更合理。

这就是本文提出 GSP 的核心动机。

---

# 11. Graph Sparse Prompting（GSP）

GSP 的核心思想：

> **不要 Prompt 所有图元素，而是自适应、稀疏地选择真正有用的元素。**

本文具体研究的是：

> **选择哪些节点特征维度值得被 Prompt。**

所以：

```text
GPF
所有特征维度都 Prompt
        ↓
GSP
只 Prompt 有用的特征维度
```

---

# 12. GSFP：Graph Sparse Feature Prompting

## 12.1 理想的 $\ell_0$ 约束

论文首先提出理想形式：

$$
\min_{\theta,p}
L_{down}
$$

满足：

$$
\tilde X=X+\mathbf1p^T
$$

以及：

$$
\boxed{\|p\|_0\le m}
$$

其中：

$$
\|p\|_0
$$

表示 $p$ 中非零元素的数量。

例如：

$$
 p=[0,2,0,-1,0,0,3]
$$

那么：

$$
\|p\|_0=3
$$

意味着只有 3 个特征维度被 Prompt。

---

# 13. 为什么不用直接优化 $\ell_0$？

因为 $\ell_0$ 是离散、非凸的，直接优化比较困难。

所以论文使用经典的稀疏正则化思想：

$$
\|p\|_0
\quad\rightarrow\quad
\|p\|_1
$$

于是目标变成：

$$
\boxed{
\min_{\theta,p}
L_{down}
+\lambda\|p\|_1
}
$$

其中：

$$
\|p\|_1=\sum_i|p_i|
$$

---

# 14. λ 到底是什么意思？

这是理解稀疏 Prompt 的关键。

目标：

$$
L_{down}+\lambda\|p\|_1
$$

其中：

- 第一项：希望任务效果好
- 第二项：希望 Prompt 尽可能稀疏

所以 $\lambda$ 是一个平衡旋钮：

```text
λ 小
↓
更重视任务损失
↓
可以使用更多 Prompt 维度

λ 大
↓
更重视稀疏性
↓
更多 p_i 被压到 0
```

论文实验也指出：

- $\lambda=0$ 时，GSFP 退化为 GPF；
- 随着 $\lambda$ 增大，Prompt 会越来越稀疏；
- 如果 $\lambda$ 太大，可能几乎没有维度被 Prompt，性能反而下降。

因此不是“越稀疏越好”，而是要找到合适的平衡。

---

# 15. GPF 与 GSFP 的本质区别

### GPF

$$
\tilde X=X+\mathbf1p^T
$$

但是：

$$
 p_1,p_2,\cdots,p_d
$$

原则上都可以非零。

### GSFP

同样：

$$
\tilde X=X+\mathbf1p^T
$$

但加入：

$$
\lambda\|p\|_1
$$

让很多：

$$
 p_i=0
$$

于是只保留少数真正有用的特征维度。

---

# 16. GSmFP：多个 Prompt

GPF-plus 使用多个 Prompt：

$$
P=[p_1,p_2,\cdots,p_k]\in\mathbb R^{d\times k}
$$

并通过 Cross-Attention 为不同节点组合这些 Prompt。

最终：

$$
\tilde X=X+SP^T
$$

其中：

- $P$：多个 Prompt
- $S$：不同节点对不同 Prompt 的权重

这比单个共享 Prompt 更灵活。

---

# 17. GSmFP 为什么使用 $\ell_{2,1}$？

目标：

> **不是让 P 中所有元素分别稀疏，而是希望整个 Prompt 行变成 0。**

定义：

$$
\boxed{
\|P\|_{2,1}
=
\sum_i\sqrt{\sum_jP_{ij}^2}
}
$$

也就是：

1. 对每一行先计算 $\ell_2$ 范数；
2. 再把所有行的 $\ell_2$ 范数加起来。

这样就会产生 **行稀疏（row sparsity）**。

---

# 18. 为什么 P 的行稀疏能选择特征？

假设：

$$
P\in\mathbb R^{d\times k}
$$

每一行对应一个原始特征维度。

例如：

$$
P=
\begin{bmatrix}
0&0\\
0.3&-0.2\\
0&0\\
0.5&0.1
\end{bmatrix}
$$

第 1、3 行全是 0。

意味着：

> 第 1、3 个特征维度不使用 Prompt。

只有第 2、4 个特征维度参与 Prompt。

所以：

$$
\boxed{
\text{P 行稀疏}
\Rightarrow
SP^T\text{ 列稀疏}
\Rightarrow
\text{选择特征维度}
}
$$

---

# 19. GSFP 与 GSmFP 的关系

| 方法 | Prompt 数量 | 稀疏方式 | 核心目标 |
|---|---:|---|---|
| GPF | 1 | 无 | 所有特征维度都可 Prompt |
| GPF-plus | 多个 | 无 | 多 Prompt + Cross-Attention |
| GSFP | 1 | $\ell_1$ | 选择少量特征维度 |
| GSmFP | 多个 | $\ell_{2,1}$ | 多 Prompt + 特征维度选择 |

可以把它理解成两条升级路线：

```text
GPF
 ↓ 加入稀疏性
GSFP
```

以及：

```text
GPF-plus
 ↓ 加入稀疏性
GSmFP
```

---

# 20. 为什么需要 Proximal Operator？

现在 GSFP 的目标是：

$$
L_{down}+\lambda\|p\|_1
$$

问题在于：

$$
|p_i|
$$

在 0 点不可微，所以不能简单地像普通参数一样使用标准梯度下降处理全部目标。

论文采用 **Forward-Backward Splitting（前向-后向分裂）**。

先处理可微的任务损失：

$$
 y=p-\eta\nabla_pL_{down}
$$

再用 Proximal Operator 处理稀疏正则项：

$$
 p\leftarrow\operatorname{Prox}^1_\lambda(y)
$$

---

# 21. $\ell_1$ 的 Proximal Operator：Soft Thresholding

定义：

$$
\operatorname{Prox}^1_\lambda(y)
=
\arg\min_z
\frac12\|z-y\|_2^2+\lambda\|z\|_1
$$

它的闭式解是：

$$
\boxed{
 z_i^*
=\operatorname{sign}(y_i)
\max(|y_i|-\lambda,0)
}
$$

也就是 **软阈值操作**。

---

# 22. Soft Thresholding 直观理解

例如：

$$
\lambda=0.2
$$

如果：

$$
 y_i=0.1
$$

那么：

$$
|y_i|<0.2
$$

所以：

$$
 z_i=0
$$

如果：

$$
 y_i=0.8
$$

那么：

$$
 z_i=0.8-0.2=0.6
$$

如果：

$$
 y_i=-0.8
$$

那么：

$$
 z_i=-0.6
$$

所以它做了两件事：

```text
小参数 → 直接压成 0
大参数 → 向 0 收缩 λ
```

这正好产生稀疏性。

---

# 23. GSFP 的训练过程

可以把 Algorithm 1 简化成：

```text
初始化 p、θ

      ↓
计算下游任务损失 Ldown
      ↓
更新 Prompt：
 p ← p − η∇p Ldown
      ↓
Soft Thresholding：
 p ← Proxλ(p)
      ↓
更新下游参数 θ：
 θ ← θ − η∇θ Ldown
      ↓
重复直到收敛
```

其中：

- $p$：稀疏 Prompt
- $\theta$：下游任务需要训练的参数，例如分类头
- 预训练 GNN 参数 $\Phi$：保持冻结

---

# 24. GSmFP 的优化

对于：

$$
L_{down}+\lambda\|P\|_{2,1}
$$

论文同样使用 Proximal Operator：

$$
P
\leftarrow
\operatorname{Prox}^{2,1}_\lambda
\left(P-\eta\nabla_PL_{down}\right)
$$

它不是逐元素 Soft Threshold，而是 **逐行进行 Group Shrinkage**。

对于第 $i$ 行：

$$
Z_i^*=
\begin{cases}
\frac{\|Y_i\|_2-\lambda}{\|Y_i\|_2}Y_i,&\lambda<\|Y_i\|_2\\
0,&\text{otherwise}
\end{cases}
$$

因此一整行可以被压成 0。

---

# 25. 今天最重要的“不要混淆”

## ① 改变输入图 ≠ 改变模型

$$
G\rightarrow\tilde G
$$

不代表：

$$
\Phi\rightarrow\Phi'
$$

Prompt Learning 通常冻结 $\Phi$。

---

## ② Prompt ≠ 新的 GNN

Prompt 是对输入数据的可学习改造，而不是重新设计一套 GNN。

---

## ③ GPF ≠ GSFP

GPF：

$$
\tilde X=X+\mathbf1p^T
$$

GSFP：

$$
\tilde X=X+\mathbf1p^T
$$

但额外加入：

$$
\lambda\|p\|_1
$$

因此 GSFP 会主动把部分 Prompt 维度压到 0。

---

## ④ $\ell_0$ ≠ $\ell_1$

$\ell_0$：直接数非零元素。

$$
\|p\|_0=\#\{i:p_i\neq0\}
$$

$\ell_1$：所有绝对值求和。

$$
\|p\|_1=\sum_i|p_i|
$$

论文用 $\ell_1$ 作为可优化的稀疏替代。

---

## ⑤ $\ell_1$ 和 $\ell_{2,1}$ 的区别

$$
\ell_1
\rightarrow
\text{逐元素稀疏}
$$

$$
\ell_{2,1}
\rightarrow
\text{逐行稀疏}
$$

在 GSmFP 中，行对应原始节点特征维度，因此行稀疏可以实现特征维度选择。

---

# 26. 用一个完整例子串起来

假设：

- 有 1000 个节点
- 每个节点 100 维特征
- 已经有一个预训练好的 GNN
- 预训练参数记为 $\Phi$
- 下游任务是节点分类

原始输入：

$$
G=(X,A)
$$

其中：

$$
X\in\mathbb R^{1000\times100}
$$

### 普通 Fine-tuning

直接修改：

$$
\Phi
$$

得到：

$$
\Phi'
$$

---

### GPF

冻结：

$$
\Phi
$$

学习：

$$
 p\in\mathbb R^{100}
$$

然后：

$$
\tilde X=X+\mathbf1p^T
$$

100 个特征维度都可以被修改。

---

### GSFP

仍然学习：

$$
 p\in\mathbb R^{100}
$$

但是增加：

$$
\lambda\|p\|_1
$$

训练后可能得到：

$$
 p=[0,0,0.3,0,0,\cdots,-0.2,0,0,0.5,\cdots]
$$

也就是说可能只有十几个维度非零。

于是：

> 模型能力没有重新训练，但输入图被有选择地调整了。

---

# 27. 把今天的内容放回 GNN 学习路线

目前可以把你已经学过的内容串成：

```text
传统图表示学习
│
├── DeepWalk
│    └── Random Walk + Skip-Gram
│
├── node2vec
│    └── 更灵活的 Random Walk
│
├── GCN
│    └── 图卷积 / 邻居聚合
│
├── GraphSAGE
│    └── 可学习的邻居聚合框架
│
├── GAT
│    └── Attention 学习不同邻居的重要性
│
└── Graph Pre-training
     └── 先学习通用图表示
          ↓
     Graph Prompt Learning
          └── 冻结预训练 GNN
               ↓
          修改输入图 / 任务形式
               ↓
          适应下游任务
               ↓
          GSP
               ├── GSFP
               └── GSmFP
```

---

# 28. 今天这篇论文真正想告诉我们的东西

不要把论文理解成：

> “作者又设计了一个新的 GNN。”

实际上并不是。

它更像是在讨论：

> **已经有一个训练好的 GNN 了，面对新的下游任务，我们怎样用尽可能少的可学习参数，让它更好地工作？**

于是出现了：

$$
\text{Fine-tuning}
\rightarrow
\text{Prompt Learning}
\rightarrow
\text{Sparse Prompt Learning}
$$

对应的思想逐渐变成：

```text
修改整个模型
      ↓
修改少量 Prompt
      ↓
只修改真正有用的 Prompt
```

这就是这篇论文的核心价值。

---

# 29. 最终知识地图

$$
\boxed{
\begin{aligned}
&\text{Pre-trained GNN}\\
&\downarrow\\
&\text{Freeze }\Phi\\
&\downarrow\\
&\text{Graph Prompt Learning}\\
&\downarrow\\
& G=(X,A)\rightarrow\tilde G=(\tilde X,A)\\
&\downarrow\\
&\text{GPF: }\tilde X=X+\mathbf1p^T\\
&\downarrow\\
&\text{发现所有特征都 Prompt 可能冗余}\\
&\downarrow\\
&\text{GSP}\\
&\quad\begin{cases}
\text{GSFP: }\ell_1\\
\text{GSmFP: }\ell_{2,1}
\end{cases}
\end{aligned}}
$$

---

# 30. 一句话记忆

> **Prompt Learning：不急着改模型，先改输入。**
>
> **GSP：不是什么都改，只改真正有用的部分。**
>
> **GSFP：用 $\ell_1$ 让单个 Prompt 稀疏。**
>
> **GSmFP：用 $\ell_{2,1}$ 让多 Prompt 按特征维度成组稀疏。**

---

## 论文来源

本文笔记根据论文 **Reliable and Compact Graph Fine-tuning via Graph Sparse Prompting** 整理，重点保留论文对 Graph Prompt Learning、GPF、GPF-plus、GSP、GSFP、GSmFP 及其优化方法的原始组织和核心公式。
