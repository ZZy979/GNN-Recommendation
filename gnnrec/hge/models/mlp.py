import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.dropout(F.relu(self.fc[i](x)))
        return self.fc[-1](x)
