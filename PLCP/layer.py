import math
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F



class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, use_bn, input_dropout, hidden_dropout,
                 node_norm):
        super().__init__()
        if nlayers == 1:
            fcs = [nn.Linear(num_features, num_classes, bias=True)]
            bns = [nn.BatchNorm1d(num_features)]
        else:
            fcs = [nn.Linear(num_features, hidden_size, bias=True)]
            bns = [nn.BatchNorm1d(num_features)]

            for i in range(nlayers - 2):
                fcs.append(nn.Linear(hidden_size, hidden_size, bias=True))
                bns.append(nn.BatchNorm1d(hidden_size))
            bns.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.Linear(hidden_size, num_classes, bias=True))

        self.fcs = nn.ModuleList(fcs)
        self.bns = nn.ModuleList(bns)
        self.input_droprate = input_dropout
        self.hidden_droprate = hidden_dropout
        self.use_bn = use_bn
        self.node_norm = node_norm
        self.reset_param()

    def reset_param(self):
        for lin in self.fcs:
            lin.reset_parameters()

    def normalize(self, embedding):
        return embedding / (1e-12 + torch.norm(embedding, p=2, dim=-1, keepdim=True))

    def forward(self, X):
        if self.node_norm:
            X = self.normalize(X).detach()
        if self.use_bn:
            X = self.bns[0](X)
        embs = F.dropout(X, self.input_droprate, training=self.training)  # .detach()
        embs = self.fcs[0](embs)

        for fc, bn in zip(self.fcs[1:], self.bns[1:]):
            embs = F.relu(embs)
            if self.node_norm:
                embs = self.normalize(embs)

            if self.use_bn:
                embs = bn(embs)
            embs = F.dropout(embs, self.hidden_droprate, training=self.training)

            embs = fc(embs)

        return embs