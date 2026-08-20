import torch
import torch.nn as nn
import random


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings.
    (Python 3.8 + Modern PyTorch Compatible Version)
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn -- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        # 【改进】统一使用 device 对象，替代分散的 .cuda() 调用
        self.device = torch.device("cuda" if cuda else "cpu")

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set

        # 【修复】采样逻辑增加空集保护
        if num_sample is not None:
            samp_neighs = []
            for to_neigh in to_neighs:
                if len(to_neigh) == 0:
                    samp_neighs.append(_set())
                elif len(to_neigh) <= num_sample:
                    samp_neighs.append(to_neigh)
                else:
                    samp_neighs.append(_set(random.sample(list(to_neigh), num_sample)))
        else:
            samp_neighs = to_neighs

        # GCN 模式：加入自身节点
        if self.gcn:
            samp_neighs = [
                samp_neigh | {nodes[i]}
                for i, samp_neigh in enumerate(samp_neighs)
            ]

        # 【修复】空邻居集合保护，避免 set.union() 在空列表上报错
        non_empty_samp = [s for s in samp_neighs if len(s) > 0]
        if len(non_empty_samp) == 0:
            # 所有节点都没有邻居，返回零向量
            embed_dim = self.features(torch.LongTensor([0]).to(self.device)).size(1)
            return torch.zeros(len(nodes), embed_dim).to(self.device)

        unique_nodes_list = list(set.union(*non_empty_samp))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # 【修复】移除 Variable，直接创建 Tensor 并指定 device
        mask = torch.zeros(len(samp_neighs), len(unique_nodes)).to(self.device)

        column_indices = [
            unique_nodes[n]
            for samp_neigh in samp_neighs
            for n in samp_neigh
        ]
        row_indices = [
            i
            for i in range(len(samp_neighs))
            for j in range(len(samp_neighs[i]))
        ]

        if len(row_indices) > 0:
            mask[row_indices, column_indices] = 1

        # 【修复】确保浮点除法，避免整除导致 mask 归零
        num_neigh = mask.sum(1, keepdim=True).float()
        # 防止除以零（孤立节点）
        num_neigh = num_neigh.clamp(min=1.0)
        mask = mask.div(num_neigh)

        # 获取邻居特征并聚合
        embed_matrix = self.features(
            torch.LongTensor(unique_nodes_list).to(self.device)
        )
        to_feats = mask.mm(embed_matrix)

        return to_feats
