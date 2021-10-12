#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx
import multiprocessing as mp
import numpy as np

def computeNewLabels(f_bar,neighborhood_avg):
    new_labels = np.zeros(neighborhood_avg.shape)
    indexes = np.argmax(neighborhood_avg - f_bar,axis=1)
    for index,col in enumerate(indexes):
        new_labels[index,col] = 1
    return new_labels

total_cpu = mp.cpu_count()
pool = mp.Pool(total_cpu)

def myGAM(G,k):
    V = G.number_of_nodes()
    indexes = np.random.randint(0,k,size=V)
    labels = np.zeros((V,k))
    for index,col in enumerate(indexes):
        labels[index,col] = 1
    t = 0
    degrees_dict = dict(G.degree())
    degrees = [degrees_dict[key] for key in degrees_dict.keys()]
    HT = {}
    HT[str(labels)] = 0
    D_inv = np.linalg.inv(np.diag(degrees))
    A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    while True:
        f = np.matmul(D_inv,A)
        f = f.dot(labels)
        f_bar = f.mean(axis=0)
        f_split = np.array_split(f, total_cpu)
        labels = np.vstack([pool.apply(computeNewLabels, args=(f_bar,row)) for row in f_split])
        if str(labels) in HT:
            return degrees, A, labels,f_split
        t = t+1
        HT[str(labels)] = t
