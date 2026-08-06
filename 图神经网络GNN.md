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