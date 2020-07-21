from torch import nn
from torch import sigmoid
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, features_count, n_hl1, n_hl2):
        super(Net, self).__init__()

        self.hidden_layer_1 = nn.Linear(features_count, n_hl1)
        self.hidden_layer_2 = nn.Linear(n_hl1, n_hl2)
        self.output_layer = nn.Linear(n_hl2, 1)

    def forward(self, X):
        h1 = F.relu(self.hidden_layer_1(X))
        h2 = F.relu(self.hidden_layer_2(h1))
        y = sigmoid(self.output_layer(h2))
        return y
