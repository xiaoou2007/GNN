import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='GAT with sparse version.')
parser.add_argument('--seed', type=int, default=72, metavar='S',
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100,
                    help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 自动选择设备
device = torch.device("cuda" if args.cuda else "cpu")

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha).to(device)
else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# 将数据移动到设备
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

if args.cuda:
    print(f'Using CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best_loss = float('inf') # 初始化一个无穷大的损失值

for epoch in range(args.epochs):
    loss_val_current = train(epoch) # train 函数返回的是 val_loss
    loss_values.append(loss_val_current)

    # 只有当当前验证集损失比历史最好损失还低时，才保存
    if loss_val_current < best_loss:
        best_loss = loss_val_current
        best_epoch = epoch
        # 只保存这一个文件，覆盖之前的
        torch.save(model.state_dict(), 'best_model.pkl')
        print(f"  -> New best model saved at epoch {epoch} with loss {best_loss:.4f}")

    # 早停逻辑
    if len(loss_values) > 1 and loss_values[-1] > loss_values[-2]:
        bad_counter += 1
        if bad_counter == args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 训练结束后，加载最好的模型进行测试
print("Optimization Finished!")
model.load_state_dict(torch.load('best_model.pkl', map_location=device))
test()

# 可选：训练完后删除这个唯一的 pkl 文件，保持目录干净
# import os
# os.remove('best_model.pkl')

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Cleanup saved models
import os
for f in os.listdir('.'):
    if f.endswith('.pkl'):
        os.remove(f)
