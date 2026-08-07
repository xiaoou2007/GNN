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