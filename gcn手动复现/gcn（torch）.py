import torch
import torch.nn.functional as F

A = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

I = torch.eye(3)
A_hat = A + I

D_hat = torch.diag(A_hat.sum(dim=1))

D_inv_sqrt = torch.diag(torch.pow(D_hat.diag(), -0.5))

A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

X = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])
W = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])
H1 = A_norm @ X @ W
H1 = F.relu(H1)
print(H1)

W2 = torch.tensor([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

H2 = A_norm @ H1 @ W2
H2 = F.relu(H2)
print(H2)

outcome = F.softmax(H2,dim=1)
labels = torch.tensor([
    0,
    1,
    0
])
loss = F.cross_entropy(H2, labels)
print(loss)