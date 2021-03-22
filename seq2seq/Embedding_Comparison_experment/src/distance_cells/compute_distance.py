import  math
import heapq


strs = []
#with open("./out1.txt", 'r') as f:

with open("./distance_cells/out1.txt", 'r') as f:
    line = f.readline().strip()
    while line:
        cs = line.split(' ')
        strs += cs
        # print (cs)
        line = f.readline().strip()

mp = {}
nodes = []
nodes.append('<s>')
nodes.append('</s>')


for s in strs:
    if mp.get(s) is not None:
        mp[s] += 1
    else:
        mp[s] = 1
        nodes.append(s)


#print("The num of nodes:", len(nodes))




#cell_name:id,such as（c1a1:0,c2a2:1....）
idx = {nodes[index]: index for index in range(len(nodes))}

#edge [i] [J] represents the minimum number of jumps between the i-th cell and the j-th cell
edge = [[int(100000) for i in nodes] for j in nodes]

#where I and j are output in the following order
for kv in idx.items():
    print (kv)
#print (idx['C16129A22460'])

#with open("./out1.txt", 'r') as f:

with open("./distance_cells/out1_train.txt", 'r') as f:
    line = f.readline().strip()
    while line:
        cs = line.split(' ')
        #strs += cs
        #Traverse all nodes
        for i in range(len(cs)):
            for j in range(len(cs)):
                idx_i = idx[cs[i]]
                idx_j = idx[cs[j]]
                dist = abs(i - j)
                edge[idx_i][idx_j] = min(edge[idx_i][idx_j], dist)

        line = f.readline().strip()


#Only the first five of the minimum distances are kept, and the rest are 100000
#todo
for i in range(len(edge)):
    edge[i][0]=100000
    edge[i][1]=100000
    edge[0][i] = 100000
    edge[1][i] = 100000
    edge[i][i]=0
edge[0][0]=0
edge[1][1]=0

param1=0.7
param2=1


for i in range(len(edge)):
    for j in range(len(edge[i])):
        #print(edge[i][j])
        if edge[i][j]>=param2+1:
            edge[i][j]=100000


#convert distance to weight
for i in range(len(edge)):
    sum_i=0
    for j in range(len(edge)):
        sum_i+=math.exp(-1*edge[i][j]/param1)
    for k1 in range(len(edge)):
        edge[i][k1]=math.exp(-1*edge[i][k1]/param1)/sum_i


for i in range(len(edge)):
    print(edge[i])
