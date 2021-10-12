import networkx as nx
import multiprocessing as mp
import numpy as np

def belonging0(f_bar,neighborhood_avg,prev_labels,cte):
	diff = np.ones(f_bar.shape[0]) - f_bar
	new_labels = neighborhood_avg*diff[np.newaxis,:]
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels

def belonging1(f_bar,neighborhood_avg,prev_labels,cte):
	diff = np.ones(f_bar.shape[0]) - f_bar
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg - f_bar[np.newaxis,:],0.0)
	new_labels *= diff[np.newaxis,:]
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels
	
def belonging2(f_bar,neighborhood_avg,prev_labels,cte):
	diff = np.ones(f_bar.shape[0]) - f_bar
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg,0.0)
	new_labels *= diff[np.newaxis,:]
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels
	
def belonging3(f_bar,neighborhood_avg,prev_labels,cte):
	new_labels = neighborhood_avg / f_bar
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels

def belonging4(f_bar,neighborhood_avg,prev_labels,cte):
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg - f_bar[np.newaxis,:],0.0)
	new_labels /= (f_bar + cte)
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels
	
def belonging5(f_bar,neighborhood_avg,prev_labels,cte):
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg,0.0)
	new_labels /= (f_bar + cte)
	row_sums = new_labels.sum(axis=1)
	new_labels /= row_sums[:, np.newaxis]
	return new_labels

def check_convergence(prev,new,epsilon,niter,maxiter):
    vec = np.linalg.norm(prev-new,axis=1)
    if any(x > epsilon for x in vec) and niter < maxiter:
        return True #informa que o algoritmo deve continuar
    return False #terminou

total_cpu = mp.cpu_count()
pool = mp.Pool(total_cpu)
def myGAM(G,k,epsilon=0.001,consider_PR=False,maxiter=100,mode=1,retry=False):
	total_rounds = 0
	V = G.number_of_nodes()
	cte = 1/(2*V)
	labels = np.zeros((V,k))
	if k != V:
		indexes = np.random.randint(0,k,size=V)
		for index,col in enumerate(indexes):
			labels[index,col] = 1
	else:
		labels = np.diag(np.ones(V))
	degrees_dict = dict(G.degree())
	degrees = [degrees_dict[key] for key in degrees_dict.keys()]
	D_inv = np.linalg.inv(np.diag(degrees))
	A = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    #A = np.where(A > 0, 1, 0) #dont consider weights
	prev_labels = labels
	not_convergence = True
	f_bar = labels.mean(axis=0)
	func = None
	if mode == 0:
		func = belonging0
	if mode == 1:
		func = belonging1
	if mode == 2:
		func = belonging2
	if mode == 3:
		func = belonging3
	if mode == 4:
		func = belonging4
	if mode == 5:
		func = belonging5

	for step in range(0,2):
		while not_convergence:
				f = np.matmul(D_inv,A)
				f = f.dot(labels)
				f_split = np.array_split(f, total_cpu)
				prev_labels_split = np.array_split(f,total_cpu)
				labels = np.vstack([pool.apply(func, args=(f_bar,row,prev_labels_row,cte)) for row,prev_labels_row in zip(f_split,prev_labels_split)])
				not_convergence = check_convergence(prev_labels,labels,epsilon,total_rounds,maxiter)
				prev_labels = labels
				f_bar = f.mean(axis=0)
				total_rounds += 1
				if total_rounds % 100 == 0:
					print(f"Took {total_rounds} rounds\n")
		print(f"Took {total_rounds} rounds\n")
		"""if step == 0 and retry == True:
					labels_unique = np.unique(np.argmax(labels, axis=1))
					labels = np.take(labels,labels_unique,axis=1)
					row_sums = labels.sum(axis=1)
					labels /= (row_sums[:, np.newaxis])
					
					not_convergence = True
					prev_labels = labels
					f_bar = labels.mean(axis=0)"""
		if step == 0 and retry == True:
					labels_unique = np.unique(np.argmax(labels, axis=1), return_inverse=True)
					n_labels = len(np.unique(labels_unique[0]))
					labels = np.zeros((V,n_labels))
					labels[np.arange(V),labels_unique[1]] = 1
					
					not_convergence = True
					prev_labels = labels
					f_bar = labels.mean(axis=0)
		else:
			break
	labels = np.where(labels > 0.01,labels,0.0)
	row_sums = labels.sum(axis=1)
	labels /= (row_sums[:, np.newaxis] + cte)
	return labels
