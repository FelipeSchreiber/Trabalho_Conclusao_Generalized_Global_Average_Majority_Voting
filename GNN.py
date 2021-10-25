import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import *
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.nn import PairwiseDistance
from torch import matmul,transpose 
from torch_geometric.nn import SAGEConv

def getJaccardMatrix(G):
	combinations = []
	for node in G.nodes():
		for key,lista in dict(nx.bfs_successors(G,node,2)).items():
			for n in lista:
				combinations.append((node,n))
	J = nx.jaccard_coefficient(G,combinations)
	V = len(G)
	simmilarity = np.zeros((V,V))
	for u, v, p in J:
	    simmilarity[u,v] = p
	simmilarity += np.diag(np.eye(len(G)))
	return simmilarity

def getNodeFeatures(G):
	degrees_dict = dict(G.degree())
	degrees = np.array([degrees_dict[key] for key in degrees_dict.keys()],dtype=np.double)
	ranks_dict = nx.algorithms.link_analysis.pagerank_alg.pagerank_numpy(G)
	ranks = np.array([ranks_dict[key] for key in ranks_dict.keys()],dtype=np.double)
	degrees /= degrees.max()
	ranks /= ranks.max()
	return degrees, ranks

def getTensorData(G):
	degrees,ranks = getNodeFeatures(G)
	for node in G.nodes():
		G.nodes[node]['degree'] = degrees[node]
		G.nodes[node]['rank'] = ranks[node]
	data = from_networkx(G,group_node_attrs=['rank','degree'])
	return data
	
class GCN(torch.nn.Module):
	def __init__(self,dimension_size,data):
		super().__init__()
		self.conv1 = GCNConv(data.num_node_features, 16)
		self.conv2 = GCNConv(16,dimension_size)
		self.dim = dimension_size

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x,training=self.training)
		x = self.conv2(x,edge_index)
		return x
		#return F.log_softmax(x, dim=1)

def train(model,data,simmilarity,epochs,min_loss=1e3):
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	pdist = PairwiseDistance(p=2)
	loss = Tensor([[1e3+1]])
	epoch = 0
	model.train()
	while Tensor.sum(loss) > min_loss and epoch < epochs:
		optimizer.zero_grad()
		out = model(data)
		out_trans = transpose(out,0,1)
		result = matmul(out,out_trans)
		#loss = pdist(result,simmilarity)
		result = binary
		loss = torch.sum(loss)
		loss.backward()
		optimizer.step()
		epoch+=1

def getEmbeddings(G,dimension=2,epochs=200):
	simmilarity = getJaccardMatrix(G)
	data = getTensorData(G)
	data.x = Tensor.float(data.x)
	simmilarity = Tensor(simmilarity)
	model = GCN(dimension,data)
	train(model,data,simmilarity,epochs)
	out = model(data).detach().numpy()
	return out
