import dgl
import torch
from torch import nn
import torch.nn.functional as F


class GRAND(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, input_droprate, hidden_droprate, dropnode_rate, order,
                 nlayers,
                 use_bn, node_norm):
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

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.order = order
        self.nlayers = nlayers
        self.use_bn = use_bn
        self.node_norm = node_norm
        self.dropnode_rate = dropnode_rate

    def reset_param(self):
        for lin in self.fcs:
            lin.reset_parameters()

    def normalize(self, embedding):
        return embedding / (1e-12 + torch.norm(embedding, p=2, dim=-1, keepdim=True))

    def rand_prop(self, features, dropnode_rate, order, graph):
        n = features.shape[0]
        drop_rate = dropnode_rate
        drop_rates = torch.full((n,), drop_rate).cuda()
        if self.training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
            features = masks * features
        else:
            features = features * (1. - drop_rate)

        propagated_features = features.clone()

        x = features.clone()
        for _ in range(order):
            graph.ndata['h'] = x
            graph.update_all(message_func=dgl.function.u_mul_e('h', 'm', 'm'), reduce_func=dgl.function.sum('m', 'h'))
            x = graph.ndata.pop('h')
            propagated_features += x
        return propagated_features.div_(order + 1.0).detach_()

    def forward(self, X, graph):
        with torch.no_grad():
            X = self.rand_prop(X, self.dropnode_rate, self.order, graph)
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
