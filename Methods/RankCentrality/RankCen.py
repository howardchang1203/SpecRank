
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import Counter
from math import factorial
import time
num_item = 30
num_clu = 2
#threshold = 0.15
#num_compare = 1000
#%% synthetic data
timestart = time.time()
#b0 = np.column_stack((b,np.zeros((2,15)))) 
#comparison = np.random.randint(num_item, size=(2, num_compare)) #data from workers
worker_num = 30
#data = np.load("C:\\Users\\Howard\\Desktop\\AISTATS\\Data\\syn\\new complete data for ave5(with feature vector)\\syn observation w from x_v5.npy")
#true_c = np.load("C:\\Users\\Howard\\Desktop\\AISTATS\\Data\\syn\\new complete data for ave5(with feature vector)\\syn true_c w from x.npy")[:,0]
#true_s = np.load("C:\\Users\\Howard\\Desktop\\AISTATS\\Data\\syn\\new complete data for ave5(with feature vector)\\syn true_s w from x.npy")
data = np.load("D:\\AISTATS\\Data\\syn\\syn w from x missing80_cover previous miss data.npy")
true_s = np.load("D:\\AISTATS\\Data\\syn\\syn true_s w from x.npy")
true_c = np.load("D:\\AISTATS\\Data\\syn\\syn true_c w from x.npy")[:,0]
#observation = []
#worker_num_ob = list(Counter(data[:,0]).values())
#k = 0
#for i in range(len(worker_num_ob)):    
#    wperworker = np.zeros((num_item,num_item))
#    for j in range(worker_num_ob[i]):
#        if data[j+k,3]==1:            
#            wperworker[data[j+k,1]-1,data[j+k,2]-1] = data[j+k,3]
#        else:
#            wperworker[data[j+k,2]-1,data[j+k,1]-1] = 1    
#    observation.append(wperworker)
#    k = k+worker_num_ob[i]
#observation = np.array(observation)
tri_upper_no_diag = np.triu(data, k=1)
for i in range(worker_num):
    for j in range(num_item):
        for k in range(num_item):
            if tri_upper_no_diag[i,j,k]==0  and k>j:
                tri_upper_no_diag[i,j,k] = 0
                tri_upper_no_diag[i,k,j] = 1
            if tri_upper_no_diag[i,j,k]==-1  and k>j:   
                tri_upper_no_diag[i,j,k] = 0
A = np.zeros((num_item,num_item))
for i in range(len(tri_upper_no_diag)):
    A = A + tri_upper_no_diag[i,:,:]
#A1 = np.zeros((num_item,num_item))
#for i in range(len(A1)):
#    for j in range(len(A1)):
#        if j == i:
#            A1[i,j] = 0
#        elif j>i:
#            A1[i,j] = A[i,j]
#        else:
#            A1[i,j] = (num_item-A[j,i])
            
B = np.zeros((num_item,num_item))
for i in range(len(A)):
    for j in range(len(A)):
        if A[i,j]==0 and A[j,i]==0:
            B[i,j] = 0
        else:
            B[i,j] = A[i,j]/(A[i,j]+A[j,i])
#B[np.isnan(B)] = 0
#C = np.zeros((num_item,num_item))
#for i in range(len(B)):
#    for j in range(len(B)):
#        if i==j:
#            C[i,j] = 0
#        elif B[i,j]==0 and B[j,i]==0:
#            C[i,j] = 0
#        else:
#            C[i,j] = np.abs(B[i,j]-0.5)
#for i in range(len(C)):
#    for j in range(len(C)):
#        if C[i,j]<threshold:
#            C[i,j] = 0



#D = np.zeros((num_item,num_item))
#%% spectral clustering
#for i in range(len(D)):
#    D[i,i] = np.sum(C[i,:])
#L = D-C
##L_rw = np.dot(np.linalg.inv(D), L)
##L_sys = np.dot(np.linalg.inv(np.sqrt(D)), L,np.linalg.inv(np.sqrt(D)))
#W_eigval, W_eigvec = np.linalg.eig(L) #eigenvectors of items
#X = []
#for i in range(num_clu):
#    X.append(W_eigvec[:,np.argsort(W_eigval)[i]])
#X = np.array(X).T
#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#esti_c = kmeans.labels_
#num_samecluster = 0
#for i in range(len(esti_c)):
#    for j in range(len(esti_c)):
#        if (j>i) and esti_c[i]==esti_c[j] and true_c[i]==true_c[j]:
#                num_samecluster = num_samecluster+1
#for i in range(len(esti_c)):
#    for j in range(len(esti_c)):
#        if (j>i) and esti_c[i]!=esti_c[j] and true_c[i]!=true_c[j]:
#                num_samecluster = num_samecluster+1
#cluster_acc =  num_samecluster/(factorial(num_item)/(factorial(2)*factorial(num_item-2)))
#%% stationary distribution

#C_new = np.dot(X,X.T)

PC = np.zeros((num_item,num_item))
for i in range(len(PC)):
    for j in range(len(PC)):
        PC[i,j] = B[j,i]    #transport to observation
d_all = []
for i in range(len(PC)):
    d_all.append(np.sum(PC[i,:]))
d_max_all = max(d_all)    
PT_all = np.zeros((num_item,num_item))
 
for i in range(len(PT_all)):
    for j in range(len(PT_all)):
        if i==j:
            PT_all[i,j] = 1-(np.sum(PC[i,:])/d_max_all)
        else:
            PT_all[i,j] = PC[i,j]/d_max_all   

def sprk(P): 
    P_trans = P.T-np.identity(len(P))
    sum_p = np.ones((len(P_trans),1)).T
    P_sol = np.vstack((P_trans,sum_p))
    b_sol = np.vstack((np.zeros((len(P_trans),1)),np.ones((1,1))))
    pi_stationary = np.linalg.lstsq(P_sol, b_sol,rcond=None)[0]   #ranking
    pi_stationary = np.reshape(pi_stationary,(len(P_trans),))
    return pi_stationary
#%%
#esti_s_all = np.zeros((num_item,))
#pi_final = sprk(PT_all)
#for i in range(num_item):
#    for j in range(num_item):
#        esti_s_all[i] = esti_s_all[i] + pi_final[j]*B[i,j]/np.sum(B[j,:])
#%%
esti_s_all = sprk(PT_all)

timeend = time.time()
print('RankClus runs:', timeend - timestart)
esti_s0 = []
esti_s1 = []
true_s0 = []
true_s1 = []
for i in range(len(np.where(true_c==0)[0])):
    true_s0.append(true_s[np.where(true_c==0)[0][i]])
    esti_s0.append(esti_s_all[np.where(true_c==0)[0][i]])
for i in range(len(np.where(true_c==1)[0])):
    true_s1.append(true_s[np.where(true_c==1)[0][i]])   
    esti_s1.append(esti_s_all[np.where(true_c==1)[0][i]])
esti_s0 = np.array(esti_s0)
esti_s1 = np.array(esti_s1)
true_s0 = np.array(true_s0)
true_s1 = np.array(true_s1)
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
r_acc0,r_sps0 = rkacc(esti_s0,true_s0)
r_acc1,r_sps1 = rkacc(esti_s1,true_s1)

#r_acc,r_sps = rkacc(sprk(PT_all),true_s) 

#%%

avg_WMW = (r_acc1+r_acc0)/2
avg_SP = (r_sps0+r_sps1)/2


