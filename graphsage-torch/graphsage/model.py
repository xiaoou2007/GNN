import os
import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
(Python 3.8 + Modern PyTorch Compatible Version)
"""

# 获取当前脚本所在目录，确保数据路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)  # xavier_uniform 已弃用，改为 xavier_uniform_

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int_)  # np.int64 -> np.int_
    node_map = {}
    label_map = {}

    content_path = os.path.join(BASE_DIR, "cora", "cora.content")
    with open(content_path) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            # 【关键修复】Python3 中 map 返回迭代器，必须转 list
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if info[-1] not in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    cites_path = os.path.join(BASE_DIR, "cora", "cora.cites")
    with open(cites_path) as fp:
        for line in fp:
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708

    feat_data, labels, adj_lists = load_cora()

    features = nn.Embedding(num_nodes, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # 【修复】统一 cuda=False，避免设备不一致报错
    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)

    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)

        start_time = time.time()
        optimizer.zero_grad()

        # 【修复】移除 Variable（PyTorch 0.4+ 不再需要）
        batch_labels = torch.LongTensor(labels[np.array(batch_nodes)])
        loss = graphsage.loss(batch_nodes, batch_labels)

        loss.backward()
        optimizer.step()

        end_time = time.time()
        times.append(end_time - start_time)

        # 【修复】loss.data[0] -> loss.item()
        print(f"Batch {batch}, Loss: {loss.item():.4f}")

    val_output = graphsage.forward(val)
    val_pred = val_output.detach().numpy().argmax(axis=1)
    print("Validation F1:", f1_score(labels[val], val_pred, average="micro"))
    print("Average batch time:", np.mean(times))


def load_pubmed():
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int_)
    node_map = {}

    node_path = os.path.join(BASE_DIR, "pubmed-data", "Pubmed-Diabetes.NODE.paper.tab")
    with open(node_path) as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])

    adj_lists = defaultdict(set)
    cites_path = os.path.join(BASE_DIR, "pubmed-data", "Pubmed-Diabetes.DIRECTED.cites.tab")
    with open(cites_path) as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    return feat_data, labels, adj_lists


def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717

    feat_data, labels, adj_lists = load_pubmed()

    features = nn.Embedding(num_nodes, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)

    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)

        start_time = time.time()
        optimizer.zero_grad()

        batch_labels = torch.LongTensor(labels[np.array(batch_nodes)])
        loss = graphsage.loss(batch_nodes, batch_labels)

        loss.backward()
        optimizer.step()

        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Batch {batch}, Loss: {loss.item():.4f}")

    val_output = graphsage.forward(val)
    val_pred = val_output.detach().numpy().argmax(axis=1)
    print("Validation F1:", f1_score(labels[val], val_pred, average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()