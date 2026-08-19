# 图神经网络GNN

## 图卷积网络GCN

| 英文                       | 中文              | 真正含义（不要记机翻）             |
| -------------------------- | ----------------- | ---------------------------------- |
| Graph                      | 图                | 节点+边组成的数据结构              |
| Node                       | 节点              | 图中的一个实体                     |
| Edge                       | 边                | 节点之间的连接关系                 |
| Graph-structured data      | 图结构数据        | 社交网络、论文引用、知识图谱等     |
| Semi-supervised Learning   | 半监督学习        | 少量标签+大量无标签数据训练        |
| Convolution                | 卷积              | 从邻居收集信息（Graph中与CNN不同） |
| Representation             | 表示（Embedding） | 神经网络学习得到的新特征           |
| Spectral Graph Convolution | 谱图卷积          | GCN之前的图卷积方法                |
| Localized                  | 局部化            | 只关注附近邻居，不关注整个图       |
| First-order Approximation  | 一阶近似          | 数学上的近似推导方法               |

| 英文                   | 中文           | 人话解释                               |
| ---------------------- | -------------- | -------------------------------------- |
| Node Classification    | 节点分类       | 预测每个节点属于什么类别               |
| Graph Regularization   | 图正则化       | 强迫相邻节点预测相似                   |
| Loss Function          | 损失函数       | 衡量预测好坏的函数                     |
| Graph Laplacian        | 图拉普拉斯矩阵 | 图学习中最重要的矩阵之一（后面重点讲） |
| Adjacency Matrix (A)   | 邻接矩阵       | 记录节点是否相连                       |
| Degree Matrix (D)      | 度矩阵         | 对角线上记录每个节点的度数             |
| Gradient Propagation   | 梯度传播       | 标签信息通过图结构传播到无标签节点     |
| Layer-wise Propagation | 分层传播       | GCN 每一层如何聚合邻居信息的规则       |

| 英文                    | 中文           | 一句话理解                       |
| ----------------------- | -------------- | -------------------------------- |
| Graph Fourier Transform | 图傅里叶变换   | 图信号分析工具，GCN 理论基础之一 |
| Chebyshev Polynomial    | 切比雪夫多项式 | 用来近似谱卷积，提高计算效率     |
| Renormalization Trick   | 重归一化技巧   | 加自环再归一化，稳定训练         |
| Self-loop               | 自环           | 节点和自己相连的一条边           |
| Propagation Rule        | 传播规则       | GCN 每层如何更新节点表示         |
| Hidden Representation   | 隐藏表示       | 每层输出的新节点特征             |
| Weight Matrix           | 权重矩阵       | GCN真正学习的参数                |
| Activation Function     | 激活函数       | ReLU等非线性函数                 |

### ⭐ 核心思想

GCN一层实际上做三件事：

收集邻居信息

↓

线性变换

↓

激活函数



| 名词                    | 作用                            | 必须掌握程度                    |
| ----------------------- | ------------------------------- | ------------------------------- |
| Spatial Convolution     | 在空间中滑动卷积核（CNN）       | ⭐⭐⭐⭐⭐                           |
| Spectral Convolution    | 在频域中定义卷积（GCN理论来源） | ⭐⭐⭐⭐⭐                           |
| Convolution Theorem     | 空间卷积 ↔ 频域乘法             | ⭐⭐⭐⭐☆（理解思想即可）           |
| Graph Fourier Transform | 图上的傅里叶变换                | ⭐⭐☆☆☆（先知道用途，不急着证明） |

Graph Laplacian 是描述图结构中节点特征变化程度的算子。它通过节点与邻居之间的差异定义“图上的平滑性”，其特征向量构成图傅里叶基，特征值对应图频率，因此成为谱图卷积的数学基础。



| 东西             | 几何理解                          | 在GCN中的作用  |
| ---------------- | --------------------------------- | -------------- |
| (L) 图拉普拉斯   | 图上的“变化检测器”                | 描述图结构     |
| (u) 特征向量     | 图上的一个基本变化模式/“图上的波” | 构成图傅里叶基 |
| (\lambda) 特征值 | 这个模式变化有多剧烈              | 对应图频率     |

                图上的节点特征
                      │
                      ↓
        ┌────────────────────────┐
        │     各种变化模式       │
        └────────────────────────┘
             ↓       ↓       ↓
          平滑变化  一般变化  剧烈变化
             ↓       ↓       ↓
           低频     中频      高频
             ↓       ↓       ↓
          λ 很小   λ 中等    λ 很大
        节点空间
           │
           │ Uᵀ
           ↓
       图傅里叶域
           │
           │ gθ
           ↓
       过滤不同频率
           │
           │ U
           ↓
        节点空间
| 普通信号处理 | 图信号处理   |
| ------------ | ------------ |
| 空间         | 图           |
| 微分算子     | 图拉普拉斯L  |
| sin/cos      | L的特征向量  |
| 频率         | 特征值λ      |
| 傅里叶变换   | 图傅里叶变换 |

> 为什么一定要找 L 进行特征分解？

一句话：

因为L是描述图结构变化的算子，它的特征向量天然对应图上的基本变化模式，而特征值对应变化速度，所以它可以像普通傅里叶中的sin/cos一样构造图傅里叶变换。



为什么需要GCN？
↓
图不像图片，没有规则卷积
↓
借助图拉普拉斯定义图傅里叶
↓
谱卷积：
UgUᵀx
↓
但是计算太贵
↓
Chebyshev近似
↓
限制K=1
↓
得到邻居聚合
↓
加self-loop
↓
归一化
↓
GCN



X
 ↓
Dropout
 ↓
XW
 ↓
Â(XW)
 ↓
ReLU
 ↓
H¹
 ↓
Dropout
 ↓
H¹W
 ↓
Â(H¹W)
 ↓
Z
 ↓
Softmax
 ↓
节点分类







## GraphSAGE

| 英文                              | 真正含义                   |
| --------------------------------- | -------------------------- |
| Inductive Representation Learning | 学习一个生成节点表示的方法 |
| Neighborhood Aggregation          | 聚合邻居信息               |
| Aggregator                        | 邻居信息融合方式           |
| Sampling                          | 限制邻居数量，提高效率     |
| Embedding                         | 节点的低维表示             |
| Graph Representation Learning     | 学习图中的隐藏特征         |
| Transductive Learning             | 只能处理已有节点           |
| Scalability                       | 能处理大规模图             |
| Message Passing                   | 节点之间的信息传播         |
| End-to-end Learning               | 从图到任务一起优化         |

| 符号           | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| $z_u$          | 节点u的embedding（GraphSAGE生成的）                          |
| $z_v$          | "正样本"节点v的embedding——v是在固定长度随机游走中与u邻近共现的节点 |
| $z_{v_n}$      | "负样本"节点的embedding——从负采样分布 $P_n$ 中抽取           |
| $\sigma$       | sigmoid函数，把点积结果压缩到(0,1)区间                       |
| $Q$            | 负样本的数量                                                 |
| $z_u^\top z_v$ | u和v两个向量的点积（dot product），衡量向量的相似度          |

                 GraphSAGE
                     │
                     ▼
              给节点 v 找邻居
                     │
                     ▼
                Sample 邻居
                     │
                     ▼
             ┌───────┴───────┐
             │               │
          Mean             Pool/LSTM
             │               │
             └───────┬───────┘
                     ▼
                聚合邻居信息
                     │
                     ▼
       自己的信息 + 邻居的信息
                     │
                     ▼
                  Linear
                     │
                     ▼
                 Activation
                     │
                     ▼
               新的 Node Embedding
| 文件名                  | 内容说明                                                     |
| ----------------------- | ------------------------------------------------------------ |
| `__init__.py`           | 包初始化文件，仅包含 Python 2/3 兼容的 `future` 导入，无实际逻辑。 |
| `inits.py`              | **权重初始化工具**。定义了 `uniform`、`glorot`、`zeros`、`ones` 等初始化函数，供各层创建参数时使用。 |
| `layers.py`             | **基础层定义**。包含 `Layer` 基类（定义了命名、日志、变量管理等通用 API）和 `Dense` 全连接层。 |
| `metrics.py`            | **评估指标**。定义了带 mask 的交叉熵损失（sigmoid/softmax）、L2 损失和准确率计算，用于处理变长图数据。 |
| `aggregators.py`        | **核心：邻居聚合器**。GraphSAGE 的核心模块，实现了多种聚合策略：<br>• `MeanAggregator`（均值聚合）<br>• `GCNAggregator`（GCN 式聚合）<br>• `MaxPoolingAggregator` / `MeanPoolingAggregator`（池化聚合）<br>• `TwoMaxLayerPoolingAggregator`（双层 MLP + 池化）<br>• `SeqAggregator`（LSTM 序列聚合） |
| `neigh_samplers.py`     | **邻居采样器**。定义 `UniformNeighborSampler`，通过从预填充的邻接表中均匀采样，为聚合器提供固定数量的邻居节点。 |
| `prediction.py`         | **边预测层**。`BipartiteEdgePredLayer` 用于计算节点对之间的亲和力分数（affinity），支持多种损失函数（交叉熵、Skip-gram、Hinge），用于无监督链接预测。 |
| `models.py`             | **模型框架与无监督模型**。包含：<br>• `Model` / `GeneralizedModel` 基类<br>• `MLP` 基准模型<br>• `SampleAndAggregate`（无监督 GraphSAGE 主模型，包含 sample + aggregate 的递归邻居聚合逻辑）<br>• `Node2VecModel`（用于对比的 Node2Vec/DeepWalk 实现） |
| `supervised_models.py`  | **监督模型**。`SupervisedGraphsage` 继承自 `SampleAndAggregate`，在节点嵌入后接 `Dense` 分类层，用于节点分类任务。 |
| `minibatch.py`          | **数据迭代器**。定义了两种 minibatch 生成器：<br>• `EdgeMinibatchIterator`：用于无监督/链接预测任务，按边采样<br>• `NodeMinibatchIterator`：用于监督/节点分类任务，按节点采样<br>同时负责构建邻接表（`construct_adj`）和划分 train/val/test。 |
| `utils.py`              | **数据加载与随机游走工具**。`load_data` 负责读取 NetworkX 图、特征、id 映射、标签等；`run_random_walks` 生成 Node2Vec 所需的节点共现对。 |
| `supervised_train.py`   | **监督训练脚本**。完整的训练流程：构建 placeholder、创建模型、训练循环、验证/测试评估（F1 分数）、日志记录。命令行入口。 |
| `unsupervised_train.py` | **无监督训练脚本**。完整的训练流程：构建 placeholder、创建模型、训练循环、验证评估（MRR 指标）、保存节点嵌入。支持 n2v 再训练模式。 |