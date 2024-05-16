import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRaLayer, self).__init__()
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim))
        self.S = torch.diag(torch.ones(rank))  # 고정된 대각 행렬

    def forward(self, x):
        weight = torch.mm(self.U, torch.mm(self.S, self.V))
        return torch.mm(x, weight)

class DeepLoRaNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank, num_layers=10):
        super(DeepLoRaNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # 중간 LoRaLayer 층
        self.middle_layers = nn.ModuleList([
            LoRaLayer(hidden_dim, hidden_dim, rank) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    in_features = 10
    hidden_size = 20
    out_features = 5

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(in_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, out_features)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MyModel()
