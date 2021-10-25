import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
import networkx as nx
from torch_geometric.utils import *
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

EPS = 1e-15

def getNodeFeatures(G):
	degrees_dict = dict(G.degree())
	degrees = np.array([degrees_dict[key] for key in degrees_dict.keys()],dtype=np.double)
	ranks_dict = nx.algorithms.link_analysis.pagerank_alg.pagerank_numpy(G)
	ranks = np.array([ranks_dict[key] for key in ranks_dict.keys()],dtype=np.double)
	degrees /= degrees.max()
	ranks /= ranks.max()
	return degrees, ranks

def getTensorData(G,mode):
	if mode == 'deg_rank':
		degrees,ranks = getNodeFeatures(G)
		for node in G.nodes():
			G.nodes[node]['degree'] = degrees[node]
			G.nodes[node]['rank'] = ranks[node]
		data = from_networkx(G,group_node_attrs=['rank','degree'])
		return data
	else:
		i = 0
		for node in G.nodes():
			vec = np.zeros(len(G))
			vec[i] = 1
			G.nodes[node]['one_hot'] = vec
			i+=1
		data = from_networkx(G,group_node_attrs=['one_hot'])
		return data

class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

def train(model,train_loader,optimizer,data):
    x, edge_index = data.x, data.edge_index
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        #adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes

def getEmbeddings(G,hidden_channels=64,epochs=200,mode='deg_rank'):
	data = getTensorData(G,mode=mode)
	data.x = Tensor.float(data.x)
	train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)
	model = SAGE(data.num_node_features, hidden_channels=hidden_channels, num_layers=2)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	for epoch in range(epochs):
		train(model,train_loader,optimizer,data)
	out = model.full_forward(data.x, data.edge_index).detach().numpy()
	return out
