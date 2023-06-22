import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import bnn

# __all__ = ["LeNet5"]


class MLPBase(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_classes):
        super(MLPBase, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class MLPBase_regression(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPBase_regression, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = x.view(-1, x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class mlp:
    base = MLPBase
    # bnn = LeNet5BNN
    args = list()
    kwargs = {"dim_in":28*28, "dim_hidden":64}

    transform_train = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )

class mlp_regression:
    base = MLPBase_regression
    args = list()
    kwargs = {"dim_hidden":10}