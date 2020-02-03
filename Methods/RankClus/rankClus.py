
from rankFunc import simpleRanking, authorityRanking
import numpy as np
from netWork import netWork
import random
import time
import heapq
import pandas as pd
from collections import defaultdict
from math import factorial
import pickle
import csv
num_item = 30
num_worker = 30
num_clu = 2
class rankClus(object):
    

    """
    """
    def __init__(self, data):
        self.network = netWork(data)
        self.pub2author, self.author2author, self.author2pub, self.publication = self.network.buildGraph()
#        global Pi
#        Pi = {}

    def initialGroup(self, K = num_clu):
        initialGroup = {i : list() for i in range(K) }
        pub = list(self.publication)
        print('publication num :', len(pub))
        for i in range(K):
            initialGroup[i].append(pub.pop(random.randint(0, len(pub) - 1)))
        pub = set(pub)
        for p in pub:
            initialGroup[random.randint(0,K - 1)].append(p)
        
        return initialGroup

    def isEmpty(self, group_list):
        result = False
        K=num_clu
        for i in range(K):
            if len(group_list[i]) == 0:
                result = True
                break
        return result
    
    def EM(self, rank_pub, rank_author, pubGroup, pub2author, author2author, author2pub, emT = 5, K = num_clu):
        
        sum_cross = float(0)
        p_k = np.zeros(K)
        sum_pub = 0.0
        for i in range(K):
            for pub in pubGroup[i]:
                for author in pub2author[pub]:
                    p_k[i] += pub2author[pub][author]
                    sum_pub += pub2author[pub][author]
        for i in range(K):
            p_k[i] /= float(sum_pub)

        new_p_k = np.zeros(K)
        while emT > 0:
            emT -= 1
            condition_p_k = {}
            for pub in pub2author:
                condition_p_k[pub] = {}
                for author in pub2author[pub]:
                    condition_p_k[pub][author] = {}

                    sum_cross += pub2author[pub][author]
                    
                    sump = float(0)
                    for k in range(K):
                        tmp = rank_pub[k][pub] * rank_author[k][author] * p_k[k]
                        sump += tmp
                        condition_p_k[pub][author][k] = tmp
                    for k in range(K):
                        condition_p_k[pub][author][k] /= sump
            
            for k in range(K):
                for pub in pub2author:
                    for author in pub2author[pub]:
                        new_p_k[k] += condition_p_k[pub][author][k] * pub2author[pub][author]

            new_p_k /= sum_cross
            p_k = new_p_k
            new_p_k = np.zeros(K)
        
        Pi = {}
        for pub in pub2author:
            normalization = float(0)
            Pi[pub] = np.zeros(K)
            for k in range(K):
                tmp = rank_pub[k][pub] * p_k[k]
                normalization += tmp
                Pi[pub][k] = tmp
            Pi[pub] /= normalization
        return Pi
#    def getpi(self):
#        return Pi

    def sim(self, x, y):
        num = np.sum(x * y)
        denom = np.linalg.norm(x) * np.linalg.norm(y)  
        cos = float(num) / denom #余弦值  
        return cos 

    def findMax(self, x, center, K = num_clu):
        max_sim = self.sim(x, center[0])
        it = 0
        for i in range(1, K):
            simt = self.sim(x, center[i])
            if simt > max_sim:
                it = i
                max_sim = simt
        return it

    def adjustGroup(self, Pi, pubGroup, K = num_clu):
        center = {}
        newGroup = {i : list() for i in range(K) }
        for i in range(K):
            center[i] = np.zeros(K)
            for pub in pubGroup[i]:
                center[i] += Pi[pub]
            center[i] /= float(len(pubGroup[i]))
        for pub in Pi:
            newGroup[self.findMax(Pi[pub], center, K)].append(pub)
        return newGroup

    """    
    iterNum: # of outer iteration
    K: # of clusters
    rankT：# of iteration of authority ranking 
    alpha：parameter of authority ranking 
    emT：# of iteration of EM alg.
    """
    def pipe(self, iterNum = 50, K = num_clu, rankT = 20, alpha = 1, emT = 10):
        time1 = time.time()
        pkl_file = open('D:\\AISTATS\\Methods\\RankClus_refined\\syn\\ini_gp_2.pkl', 'rb')
        group = pickle.load(pkl_file)
        pkl_file.close()
#        group = self.initialGroup(K)
        ini_gp = group.copy()
        time2 = time.time()
        print('Initial finished:', time2 - time1, 's')
        rank_pub = {}
        rank_author = {}
        
        iters = 0
        d = []
        while iters < iterNum:
            time1 = time.time()
            
            print('authorityRanking start:')
            for i in range(K):
                #rank_pub[i], rank_author[i], tmp= authorityRanking(group[i], self.pub2author, self.author2author, self.author2pub, rankT, alpha)
                rank_author[i], rank_pub[i],tmp= authorityRanking(self.author2pub, self.pub2author, self.author2author,group[i],  rankT, alpha)             
            time3 = time.time()
            print('authorityRanking end:', time3 - time1)
            
            Pi = self.EM(rank_pub, rank_author, group, self.pub2author, self.author2author,self.author2pub, emT, K)
#            print('Pi', Pi)
            time4 = time.time()
            print('EM :', time4 - time3)
            new_group = self.adjustGroup(Pi, group, K)
            del group
            group = new_group
            if self.isEmpty(group):
                group = self.initialGroup(K)
                print('Empty group !')
                iters = 0
            else:
                iters += 1
            time2 = time.time()
            print('Do clustering at epoch :', iters ,time2 - time1)
            d.append(group)
        a = []
        b = []
        c = []
        for i in range(K):
            rank_author, rank_pub, rank_pub_part = authorityRanking(self.author2pub, self.pub2author, self.author2author,group[i],  rankT, alpha)
            
            top_10_pub = heapq.nlargest(10, rank_pub_part.items(), lambda x: x[1])
            print('Group  '+str(i))
            for confer in top_10_pub:
                print(confer[0])
            top_10_author = heapq.nlargest(10, rank_author.items(), lambda x: x[1])
            print('* * * * * * *')
            print('worker')
            for author in top_10_author:
                print(author[0])
            print('- - - - - - - - - - - - - - - - - ')   
            a.append(rank_pub)
            b.append(rank_author) 
            c.append(top_10_pub)
        return rank_pub,d,ini_gp
#%%
timestart = time.time()
test = rankClus('test_syn_80.txt')
a,b,ini_gp = test.pipe()
#output = open('ini_gp_2.pkl', 'wb')
#pickle.dump(ini_gp, output)
#output.close()
esti_c = np.zeros((num_item,))
for i in range(len(b[-1])):
    for j in range(len(b[-1][i])):
         esti_c[int(b[-1][i][j]),] = i
    
aa = pd.DataFrame.from_dict(a, orient='index')
aa.index = aa.index.astype(int)
esti_s = np.reshape(aa.sort_index().values,(num_item,))

timeend = time.time()
print('RankClus runs:', timeend - timestart)
#%% 
true_s = np.load("D:\\AISTATS\\Data\\syn\\syn true_s w from x.npy")
true_c = np.load("D:\\AISTATS\\Data\\syn\\syn true_c w from x.npy")[:,0]
#%% cluster
num_samecluster = 0
for i in range(len(esti_c)):
    for j in range(len(esti_c)):
        if (j>i) and esti_c[i]==esti_c[j] and true_c[i]==true_c[j]:
                num_samecluster = num_samecluster+1
for i in range(len(esti_c)):
    for j in range(len(esti_c)):
        if (j>i) and esti_c[i]!=esti_c[j] and true_c[i]!=true_c[j]:
                num_samecluster = num_samecluster+1
cluster_acc =  num_samecluster/(factorial(num_item)/(factorial(2)*factorial(num_item-2)))
#%% ranking 
def rkacc(esti,true):
    true_rank = np.argsort(-true)
    estimate_rk = np.argsort(-esti)
    rank_deno = 0
    rank_num = 0
    for i_ind in range(len(true_rank)):
           for j_ind in range(len(true_rank)):
               if i_ind != j_ind: 
                   if np.where(true_rank==i_ind)[0][0] < np.where(true_rank==j_ind)[0][0]:
                       rank_deno = rank_deno + 1
                       if np.where(estimate_rk==i_ind)[0][0] < np.where(estimate_rk==j_ind)[0][0]:
                           rank_num = rank_num + 1
    if rank_deno == 0:
        rank_deno=-1
    r_acc = rank_num/rank_deno    
    sp_count=0
    for i in range(len(true)):
        sp_count=sp_count+np.abs(np.where(true_rank==i)[0][0]-np.where(estimate_rk==i)[0][0])
    return r_acc,sp_count

esti_s0 = []
esti_s1 = []
esti_s2 = []
esti_s3 = []
true_s0 = []
true_s1 = []
true_s2 = []
true_s3 = []
for i in range(len(np.where(true_c==0)[0])):
    true_s0.append(true_s[np.where(true_c==0)[0][i]])
    esti_s0.append(esti_s[np.where(true_c==0)[0][i]])
for i in range(len(np.where(true_c==1)[0])):
    true_s1.append(true_s[np.where(true_c==1)[0][i]])   
    esti_s1.append(esti_s[np.where(true_c==1)[0][i]])
#for i in range(len(np.where(true_c==2)[0])):
#    true_s2.append(true_s[np.where(true_c==2)[0][i]])   
#    esti_s2.append(esti_s[np.where(true_c==2)[0][i]])
#for i in range(len(np.where(true_c==3)[0])):
#    true_s3.append(true_s[np.where(true_c==3)[0][i]])   
#    esti_s3.append(esti_s[np.where(true_c==3)[0][i]])
esti_s0 = np.array(esti_s0)
esti_s1 = np.array(esti_s1)
esti_s2 = np.array(esti_s2)
esti_s3 = np.array(esti_s3)
true_s0 = np.array(true_s0)
true_s1 = np.array(true_s1)
true_s2 = np.array(true_s2)
true_s3 = np.array(true_s3)
r_acc0,r_sps0 = rkacc(esti_s0,true_s0)
r_acc1,r_sps1 = rkacc(esti_s1,true_s1)
#r_acc2,r_sps2 = rkacc(esti_s2,true_s2)
#r_acc3,r_sps3 = rkacc(esti_s3,true_s3)

avg_WMW = (r_acc0+r_acc1)/2
avg_SP = (r_sps0+r_sps1)/2
