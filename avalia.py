import itertools
import numpy as np
import networkx as nx
from networkx.generators.community import planted_partition_graph
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from myGAMDIV import *

#funcao que organiza o output do algoritmo retornando um dicionario onde cada entrada eh uma comunidade 
#(lista de vertices)
def generateOutput(G,l):
    communities = {}
    k = l.shape[1]
    for node in G:
        for community in range(k):
            if l[int(node)-1,community] > 0:
                if community not in list(communities.keys()):
                    communities[community] = []
                communities[community].append(int(node))
    return communities

#Funcao que recebe um dicionario e escreve em cada linha uma lista de vertices representando a comunidade    
def writeCommunityDictInFile(file,communities_dict):
    with open(file, 'w') as f:
        for key in communities_dict.keys():
            for item in communities_dict[key]:
                f.write("%s " % item)
            f.write("\n")
            
# para os benchmarks do NetworkX            
def generateGroundTruthNETX(true_values):
    communities = {}
    for node,k in true_values:
        if k not in communities.keys():
            communities[k] = []
        communities[k].append(node)    
    return communities            
         
#Para o benchmark LFR            
def generateGroundTruth(filepath):
    communities = {}
    cur_node = 1
    with open(filepath,"r") as f:
        lines = f.read().splitlines()
        for line in lines:
            line = line.split("\t")[1]
            groups = line.split(" ")
            for k in groups:
                if k != "":
                    if k not in communities.keys():
                        communities[k] = []
                    communities[k].append(cur_node)    
            cur_node += 1
    return communities

#Funcao que dado um grafo cujos vertices possuem labels inteiros [0,1,2...,N] retornado pelo SBM, retorna o grau de pertencimento de cada vértice em uma classe, que é dado pelo número de arestas intraclasse dividido pelo seu grau
def getBelongings(G):
    node_belonging = []
    for node in G.nodes():
        kin = total = 0
        node_community = G.nodes[node]["block"]
        for n in G.neighbors(node):
            total += 1
            if G.nodes[n]["block"] == node_community:
                kin += 1
        node_belonging.append(kin/total)
    return node_belonging
    
#Funcao que dada a matriz de probabilidades e o grau de pertencimento real, faz o scatter plot
def makeStructuralPlot(preds,node_belongings,filename):
	for pred,node_belonging in zip(preds,node_belongings):
		structure_found = np.max(pred,axis=1)
		plt.figure(figsize=(8,8))
		plt.scatter(node_belonging,structure_found)
	plt.xlabel("Pertencimento real")
	plt.ylabel("Pertencimento retornado pelo algoritmo")
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.savefig(filename)
	plt.clf()

#Funcao que dada a lista de comunidades retornada pelo CFinder, retorna os labels de cada vértice	
def addCommunityLabels(G,c):
    for i,partition in enumerate(c):
        for node in list(partition):
            if 'labels' not in G.nodes[node]:
                G.nodes[node]['labels'] = []
            G.nodes[node]['labels'].append(i)

#Funcao que dada a lista de comunidades retornada pelo CFinder, retorna o grau de pertencimento de cada vértice    
def getKCliqueBelongings(G,c):
	node_belonging = []
	addCommunityLabels(G,c)
	for node in G.nodes():
		neighborhood_sum = np.zeros(len(c))
		for label in G.nodes[node]["labels"]:
			neighborhood_sum[label] = 1
		for n in G.neighbors(node):
			for label in G.nodes[n]["labels"]:
				neighborhood_sum[label] += 1
		neighborhood_sum /= np.linalg.norm(neighborhood_sum)		
		node_belonging.append(neighborhood_sum)
	return np.vstack(node_belonging)
       
#Funcao que dada a matriz de probabilidades retornada pelo algoritmo e o groundtruth computa a acurácia
def acc(x,y):
    x = np.argmax(x,axis=1)
    x = np.unique(x,return_inverse=True)[1]
    if len(x) != len(y):
        raise ValueError('x and y must be arrays of the same size')
    N = len(x)
    scores = []
    possible_combinations = list(itertools.permutations(np.unique(x)))
    for combination in possible_combinations:
        pred = np.array([combination[i] for i in x])
        scores.append(accuracy_score(y,pred))
    score = max(scores)
    return score

#Funcao que gera um grafo com 1000 vertices em cada comunidade e retorna os pertencimentos reais
def generateGraph(k,pin,pout,seed):
    G = planted_partition_graph(k,1000,pin,pout,seed=seed)
    groundTruth = [data["block"] for node,data in dict(G.nodes.data()).items()]
    belonging = getBelongings(G)
    return G, groundTruth, belonging
    
def generateNGraphs(N,k,pin,pout):
	truths = []
	graphs = []
	belongings = []
	for i in range(N):
		G, truth, belonging = generateGraph(k,pin,pout,i)
		graphs.append(G)
		truths.append(truth)
		belongings.append(belonging)
	return graphs, truths, np.hstack(belongings)
    
def makeTest(k,graphs,truths,belongings,mode,retry,pout):
	results = []
	acuracias = []
	max_dim = 0
	count = 0
	for i,G in enumerate(graphs):
		l = myGAM(G,k,0.001,maxiter=30,mode=mode,retry=retry)
		#results.append(l)
		acuracias.append(acc(l,truths[i]))
		labels_max = np.argmax(l,axis=1)
		values, counts = np.unique(labels_max, return_counts=True)
		if len(values) == 1:
			count += 1
	#if retry:
	#	makeStructuralPlot(results,belongings,f"Retry_Belongings_com_{k}_labels_mode_{mode}_pout_{pout}.png")
	#else:
	#	makeStructuralPlot(results,belongings,f"Belongings_com_{k}_labels_mode_{mode}_pout_{pout}.png")
		#print(values,counts,acuracias[-1])
        #if len(values) == 1:
        #    count += 1
        #if l.shape[1] > max_dim:
        #    max_dim = l.shape[1]
        #print("mode = ",mode," shape: ",l.shape)
        #for i,arr in enumerate(results):
        #    if arr.shape[1] < max_dim:
        #        diff = max_dim - arr.shape[1]
        #        results[i] = np.hstack([arr,np.zeros((len(graphs[0]),diff))])
    #results = np.vstack(results)
	return np.mean(acuracias), np.std(acuracias), count

def makeTestChangingP(pin,pout_min,pout_max,N,k,k_init,retry,modes):
	pouts = np.arange(pout_min,pout_max,0.01)
	data = []
	for i,pout in enumerate(pouts):
		graphs, truths, belongings = generateNGraphs(N,k,pin,pout)
		for mode in modes:
			y, err, times_failed = makeTest(k_init,graphs,truths,belongings,mode,retry,pout)
      	  #if retry:
          #	makeStructuralPlot(results,belongings,f"Retry_Belongings_com_{k}_labels_mode_{mode}_pin_{pin}_pout_{pout}.png")
          #else:
          #  makeStructuralPlot(results,belongings,f"Belongings_com_{k}_labels_mode_{mode}_pin_{pin}_pout_{pout}.png")
			data.append({'mode':mode,'acc':y,'std':err,'k':k,'retry':retry,'pout':pout,'failed':times_failed})
		print(i/len(pouts),pout)
	return data	

