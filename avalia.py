import itertools
from sklearn.metrics import accuracy_score

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
    
#Funcao que dada a matriz de propabilidades retornada pelo algoritmo e o groundtruth computa a acurácia
def acc(x,y):
  	x = np.argmax(x,axis=1)
  	if len(x) != len(y):
		raise ValueError('x and y must be arrays of the same size')
	N = len(x)
	scores = []
	possible_combinations = list(itertools.permutations([np.unique(x)]))
	for combination in possible_combinations:
    	pred = np.array([combination[i] for i in x])
    	scores.append(accuracy_score(y,pred))	
    score = max(scores)
    return score     
