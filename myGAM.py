#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ALGORITMO BASEADO NO COPRA
"""
import networkx as nx
import multiprocessing as mp
import numpy as np


def belonging(f_bar, neighborhood_avg):
    new_labels = np.where(neighborhood_avg > f_bar, neighborhood_avg, 0)
    for node in range(new_labels.shape[0]):
        if np.sum(new_labels[node, :]) == 0:
            col = np.argmax(new_labels[node, :] - f_bar, axis=1)
            new_labels[node, col] = 1
        new_labels[node, :] = new_labels[node, :] \
            / np.sum(new_labels[node, :])
    return new_labels


def check_convergence(
    prev,
    new,
    epsilon,
    niter,
    maxiter,
    ):
    vec = np.linalg.norm(prev - new, axis=1)
    if any(x > epsilon for x in vec) and niter < maxiter:
        return True  # informa que o algoritmo deve continuar
    return False  # terminou


total_cpu = mp.cpu_count()
pool = mp.Pool(total_cpu)


def myGAMCopra(
	G,
	k,
	epsilon=0.001,
	maxiter=100,
	):
	total_rounds = 0
	V = G.number_of_nodes()
	labels = np.zeros((V, k))
	if k != V:
		indexes = np.random.randint(0, k, size=V)
		for (index, col) in enumerate(indexes):
			labels[index, col] = 1
	else:
		labels = np.diag(np.ones(V))
	t = 0
	degrees_dict = dict(G.degree())
	degrees = [degrees_dict[key] for key in degrees_dict.keys()]
	D_inv = np.linalg.inv(np.diag(degrees))
	A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    # A = np.where(A > 0, 1, 0) #dont consider weights
	cte = 1/(2*V)
	prev_labels = labels
	not_convergence = True
	for step in range(0, 2):
		while not_convergence:
			f = np.matmul(D_inv, A)
			f = f.dot(labels)
			f_bar = f.mean(axis=0)
			f_split = np.array_split(f, total_cpu)
			labels = np.vstack([pool.apply(belonging, args=(f_bar,row)) for row in f_split])
			not_convergence = check_convergence(prev_labels, labels,epsilon, total_rounds, maxiter)
			prev_labels = labels
			total_rounds += 1
		print ('Took {total_rounds} rounds'.format(total_rounds=total_rounds))
		if step == 0:
			labels_unique = np.unique(np.argmax(labels, axis=1))
			labels = np.take(labels, labels_unique, axis=1)
			row_sums = labels.sum(axis=1)
			labels /= (row_sums[:, np.newaxis] + cte)
			not_convergence = True
			prev_labels = labels
			f_bar = labels.mean(axis=0)
	return labels
