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
	new_labels /= (row_sums[:, np.newaxis] + cte)
	return new_labels
	
def belonging2(f_bar,neighborhood_avg,prev_labels,cte):
	diff = np.ones(f_bar.shape[0]) - f_bar
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg,0.0)
	new_labels *= diff[np.newaxis,:]
	row_sums = new_labels.sum(axis=1)
	new_labels /= (row_sums[:, np.newaxis] + cte)
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
	new_labels /= (row_sums[:, np.newaxis] + cte)
	return new_labels
	
def belonging5(f_bar,neighborhood_avg,prev_labels,cte):
	new_labels = np.where(neighborhood_avg > f_bar,neighborhood_avg,0.0)
	new_labels /= (f_bar + cte)
	row_sums = new_labels.sum(axis=1)
	new_labels /= (row_sums[:, np.newaxis]+cte)
	return new_labels
	
def belonging6(f_bar, neighborhood_avg,prev_labels,cte):
    new_labels = np.where(neighborhood_avg > f_bar, neighborhood_avg, 0.0)
    for node in range(new_labels.shape[0]):
        if np.sum(new_labels[node, :]) == 0:
            col = np.argmax(new_labels[node, :] - f_bar, axis=1)
            new_labels[node, col] = 1
        new_labels[node, :] = new_labels[node, :] \
            / np.sum(new_labels[node, :])
    return new_labels
    
def belonging7(f_bar, neighborhood_avg,prev_labels,cte):
    new_labels = np.where(neighborhood_avg > f_bar, neighborhood_avg - f_bar, 0)
    for node in range(new_labels.shape[0]):
        if np.sum(new_labels[node, :]) == 0:
            col = np.argmax(new_labels[node, :] - f_bar, axis=1)
            new_labels[node, col] = 1
        new_labels[node, :] = new_labels[node, :] \
            / np.sum(new_labels[node, :])
    return new_labels

def check_convergence(prev,new,epsilon,niter,maxiter):
    vec = np.linalg.norm(prev-new,axis=1)
    if any(x > epsilon for x in vec) and niter < maxiter:
        return True #informa que o algoritmo deve continuar
    return False #terminou

total_cpu = mp.cpu_count()
pool = mp.Pool(total_cpu)
def myGAM(G,k,epsilon=0.001,maxiter=100,mode=1,retry=False):
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
	A = A.astype(np.float16)
	A += np.eye(V)
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
	if mode == 6:
		func = belonging6
	if mode == 7:
		func = belonging7
			
	D_inv_A = np.matmul(D_inv,A)	
	for step in range(0,2):
		while not_convergence:
				f = D_inv_A.dot(labels)
				f_split = np.array_split(f, total_cpu)
				prev_labels_split = np.array_split(f,total_cpu)
				labels = np.vstack([pool.apply(func, args=(f_bar,row,prev_labels_row,cte)) for row,prev_labels_row in zip(f_split,prev_labels_split)])
				not_convergence = check_convergence(prev_labels,labels,epsilon,total_rounds,maxiter)
				prev_labels = labels
				f_bar = f.mean(axis=0)
				total_rounds += 1
				#print(f"Shape: {np.unique(np.argmax(labels, axis=1))}")
				if total_rounds % 100 == 0:
					print(f"Took {total_rounds} rounds\n")
		print(f"Took {total_rounds} rounds\n")
		if step == 0 and retry == True:
			labels_max = np.unique(np.argmax(labels, axis=1), return_inverse=True)
			n_labels = labels_max[0].shape[0]
			labels = np.zeros((V,n_labels))
			labels[np.arange(V),labels_max[1]] = 1
			not_convergence = True
			prev_labels = labels
			f_bar = labels.mean(axis=0)
		else:
			break
	labels_max = np.unique(np.argmax(labels, axis=1), return_inverse=True)
	labels = labels[:,labels_max[0]]
	row_sums = labels.sum(axis=1)
	labels /= (row_sums[:, np.newaxis] + cte)
	return labels
