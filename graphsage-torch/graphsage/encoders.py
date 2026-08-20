import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach.
    (Python 3.8 + Modern PyTorch Compatible Version)
    """

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample

        # 【修复】!= None -> is not None
        if base_model is not None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        # 【改进】统一使用 device 对象，替代分散的 .cuda() 和属性覆盖
        self.device = torch.device("cuda" if cuda else "cpu")

        # 【修复】不再直接覆盖 aggregator.cuda，device 已在 aggregator 内部统一管理
        # self.aggregator.cuda = cuda  # 已移除

        input_dim = self.feat_dim if self.gcn else 2 * self.feat_dim
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, input_dim)
        )
        # 【修复】xavier_uniform -> xavier_uniform_（原地操作版本）
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes -- list of nodes
        """
        # 获取邻居特征（aggregator 内部已处理设备）
        neigh_feats = self.aggregator.forward(
            nodes,
            [self.adj_lists[int(node)] for node in nodes],
            self.num_sample
        )

        if not self.gcn:
            # 【修复】在正确设备上创建 Tensor，替代 .cuda() 硬编码
            self_feats = self.features(
                torch.LongTensor(nodes).to(self.device)
            )
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        # weight.mm(combined.t()) 等价于 linear(combined)，保持原始逻辑
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
