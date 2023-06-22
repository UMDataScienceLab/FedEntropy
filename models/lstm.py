import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import bnn

# __all__ = ["LeNet5"]


class LSTMBase(nn.Module):
    def __init__(self, len_seq, dim_in_feature, dim_hidden, dim_output, 
                 num_layers=3, num_groups=20, init_log_noise=None, norm='batch'):
        super(LSTMBase, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_in_feature = dim_in_feature
        self.num_groups = num_groups
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        
        self.lstm = nn.LSTM(input_size=dim_in_feature, hidden_size=dim_hidden[0], batch_first=True, num_layers=num_layers)    
        if norm == 'batch':
            self.lstm_bn = nn.BatchNorm1d(dim_hidden[0]*len_seq)
        elif norm == 'group':
            self.lstm_bn = nn.GroupNorm(
                num_groups=self.num_groups, num_channels=dim_hidden[0]*len_seq
            )
        self.layer_hidden1 = nn.Linear(dim_hidden[0]*len_seq, dim_hidden[1], bias=True)
        if norm == 'batch':
            self.fc1_bn = nn.BatchNorm1d(dim_hidden[1])
        elif norm == 'group':
            self.fc1_bn = nn.GroupNorm(num_groups=self.num_groups, 
                num_channels=dim_hidden[1]
            )
        self.layer_hidden2 = nn.Linear(dim_hidden[1], dim_output, bias=True)
        
        if init_log_noise is not None:
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))
        
        
    def forward(self, x):
        
        # input shape info: (batchsize, len seq, num feature)
        batchsize = x.shape[0]
        # if self.dim_in_feature == 1:
        #     x = x.view(x.shape[0], x.shape[1], 1)
        
        # x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        # x = x.transpose(0, 1)
        x = x.reshape((batchsize, -1))
        x = self.lstm_bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.layer_hidden1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.layer_hidden2(x)

        return x

class LSTMBase2(nn.Module):
    def __init__(self, len_seq, dim_in_feature, dim_hidden, dim_output, 
                 num_layers=3, num_groups=20, init_log_noise=None, norm='batch'):
        super(LSTMBase2, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_in_feature = dim_in_feature
        self.num_groups = num_groups
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        
        self.lstm = nn.LSTM(input_size=dim_in_feature, hidden_size=dim_hidden[0], batch_first=True, num_layers=num_layers)    
        if norm == 'batch':
            self.lstm_bn = nn.BatchNorm1d(dim_hidden[0]*len_seq)
        elif norm == 'group':
            self.lstm_bn = nn.GroupNorm(
                num_groups=self.num_groups, num_channels=dim_hidden[0]*len_seq
            )
        self.layer_hidden1 = nn.Linear(dim_hidden[0]*len_seq, dim_hidden[1], bias=True)
        if norm == 'batch':
            self.fc1_bn = nn.BatchNorm1d(dim_hidden[1])
        elif norm == 'group':
            self.fc1_bn = nn.GroupNorm(num_groups=self.num_groups, 
                num_channels=dim_hidden[1]
            )
        self.layer_hidden2 = nn.Linear(dim_hidden[1], dim_output, bias=True)

        self.layer_hidden3 = nn.Linear(len_seq, dim_hidden[2], bias=True)
        self.layer_hidden4 = nn.Linear(dim_hidden[2], dim_hidden[3], bias=True)
        self.layer_hidden5 = nn.Linear(dim_hidden[3], dim_output, bias=True)
        
        if init_log_noise is not None:
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]))
        
        
    def forward(self, x):
        
        # input shape info: (batchsize, len seq, num feature)
        batchsize = x.shape[0]
        # if self.dim_in_feature == 1:
        #     x = x.view(x.shape[0], x.shape[1], 1)
        
        x2 = self.layer_hidden3(x.reshape((batchsize, -1)))
        x2 = self.relu(x2)
        x2 = self.layer_hidden4(x2)
        x2 = self.relu(x2)
        x2 = self.layer_hidden5(x2)
        
        # x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        # x = x.transpose(0, 1)
        x = x.reshape((batchsize, -1))
        x = self.lstm_bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.layer_hidden1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.layer_hidden2(x) + x2

        return x
    
    
class FixedLSTM:
    base = LSTMBase
    args = list()
    kwargs = {"dim_hidden":[6,30]}

class FixedLSTM2:
    base = LSTMBase2
    args = list()
    kwargs = {"dim_hidden":[6,30, 100]}