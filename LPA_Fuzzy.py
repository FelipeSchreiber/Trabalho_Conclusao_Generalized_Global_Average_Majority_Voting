import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import descendants_at_distance
import multiprocessing as mp
import numpy as np

def get_vectorized_transform(d):
	def func(x):
		return np.max([1 - x/(d+1),0])
	transform_dist = np.vectorize(func)
	return transform_dist

def getDistances(G,source,distance):
		V = G.number_of_nodes()
		dist_to_nodes = np.ones(V)*V
		dist_to_nodes[source] = 0
		if not G.has_node(source):
			raise nx.NetworkXError(f"The node {source} is not in the graph.")
		current_distance = 0
		queue = {source}
		visited = {source}
		# this is basically BFS, except that the queue only stores the nodes at
		# current_distance from source at each iteration
		while queue:
			current_distance += 1
			if current_distance == distance + 1:
				queue = {}
			next_vertices = set()
			for vertex in queue:
				for child in G[vertex]:
					if child not in visited:
						visited.add(child)
						dist_to_nodes[child] = current_distance
						next_vertices.add(child)
			queue = next_vertices
		return dist_to_nodes

def LPA_Fuzzy(G,d):
		"""
		Método que dada uma distancia "d", obtem um vetor para z para cada vértice u da rede tal que z[v] = 0 se o vértice v está a uma distância maior que "d" a partir de u, ou z[v] = y sendo y dado por y = 1 - x/d, onde x representa a distancia que v está de u.
		"""
		total_cpu = mp.cpu_count()
		pool = mp.Pool(total_cpu)
		distances = np.vstack([pool.apply(getDistances, args=(G,node,d)) for node in G.nodes()])
		func = get_vectorized_transform(d)
		distances = distances.astype(np.float16)
		distances = func(distances)
		return distances
