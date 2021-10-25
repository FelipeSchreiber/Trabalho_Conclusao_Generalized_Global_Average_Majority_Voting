#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import multiprocessing as mp
import numpy as np

def computeNewLabels(f_bar,neighborhood_avg):
	new_labels = np.zeros(neighborhood_avg.shape)
	relative_neighborhood = neighborhood_avg - f_bar[np.newaxis,:]
	for i in range(new_labels.shape[0]):
		indexes = np.argwhere(relative_neighborhood[i,:] == np.amax(relative_neighborhood[i,:]))
		index = np.random.choice(indexes)
		new_labels[i,index] = 1
	return new_labels

def myGAM(G,k):
    total_cpu = mp.cpu_count()
    pool = mp.Pool(total_cpu)
    V = G.number_of_nodes()
    indexes = np.random.randint(0,k,size=V)
    labels = np.zeros((V,k))
    for index,col in enumerate(indexes):
        labels[index,col] = 1
    t = 0
    HT = {}
    HT[str(labels)] = 0
    D_inv = np.diag([1/G.degree(i) for i in G])
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    D_inv_A = np.matmul(D_inv,A)
    while True:
        f = D_inv_A
        f = f.dot(labels)
        f_bar = f.mean(axis=0)
        f_split = np.array_split(f, total_cpu)
        labels = np.vstack([pool.apply(computeNewLabels, args=(f_bar,row)) for row in f_split])
        if str(labels) in HT:
            return labels
        t = t+1
        HT[str(labels)] = t
