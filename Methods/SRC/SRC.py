
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import Counter
from math import factorial
num_item = 30
num_clu = 2
threshold = 0
#num_compare = 1000
#%% synthetic data

#b0 = np.column_stack((b,np.zeros((2,15)))) 
#comparison = np.random.randint(num_item, size=(2, num_compare)) #data from workers
worker_num = 30
data = np.load("D:\\AISTATS\\Data\\syn\\syn observation w from x.npy")
true_s = np.load("D:\\AISTATS\\Data\\syn\\syn true_s w from x.npy")
true_c = np.load("D:\\AISTATS\\Data\\syn\\syn true_c w from x.npy")[:,0]
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
for i in range(len(B)):
    for j in range(len(B)):
            if A[i,j]==0 and A[j,i]==0:
                B[i,j] = 0
            else:
                B[i,j] = A[i,j]/(A[i,j]+A[j,i])
                
#B[np.isnan(B)] = 0
C = np.zeros((num_item,num_item))
for i in range(len(B)):
    for j in range(len(B)):
        if i==j:
            C[i,j] = 0
        elif B[i,j]==0 and B[j,i]==0:
            C[i,j] = 0
        else:
            C[i,j] = np.abs(B[i,j]-0.5)
for i in range(len(C)):
    for j in range(len(C)):
        if C[i,j]<threshold:
            C[i,j] = 0

D = np.zeros((num_item,num_item))
#%% spectral clustering
for i in range(len(D)):
    D[i,i] = np.sum(C[i,:])
L = D-C
#L_rw = np.dot(np.linalg.inv(D), L)
L_sys = np.dot(np.linalg.inv(np.sqrt(D)), L,np.linalg.inv(np.sqrt(D)))
W_eigval, W_eigvec = np.linalg.eig(L_sys) #eigenvectors of items
X = []
for i in range(num_clu):
    X.append(W_eigvec[:,np.argsort(W_eigval)[i]])
X = np.array(X).T
X_normal = np.zeros((len(X),num_clu))
for i in range(len(X_normal)):
    for j in range(len(X_normal[i])):
        X_normal[i,j] = X[i,j]/np.linalg.norm(X[i,:])
    
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_normal)
esti_c = kmeans.labels_
#true_c = np.load("C:\\Users\\Howard\Desktop\\EM_and_synthetic data\\0730\\true_c_lastfm_2cluster.npy")
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
#%% stationary distribution
C_new = np.dot(X_normal,X_normal.T)
C0 = np.zeros((len(np.where(esti_c==0)[0]),len(np.where(esti_c==0)[0])))
B0 = np.zeros((len(np.where(esti_c==0)[0]),len(np.where(esti_c==0)[0])))
C1 = np.zeros((len(np.where(esti_c==1)[0]),len(np.where(esti_c==1)[0])))
B1 = np.zeros((len(np.where(esti_c==1)[0]),len(np.where(esti_c==1)[0])))
clu_0_idx = np.where(esti_c==0)[0]
clu_1_idx = np.where(esti_c==1)[0]
for i in range(len(clu_0_idx)):
    for j in range(len(clu_0_idx)):
        C0[i,j] = C_new[clu_0_idx[i],clu_0_idx[j]]
        B0[i,j] = B[clu_0_idx[i],clu_0_idx[j]]
for i in range(len(clu_1_idx)):
    for j in range(len(clu_1_idx)):
        C1[i,j] = C_new[clu_1_idx[i],clu_1_idx[j]]   
        B1[i,j] = B[clu_1_idx[i],clu_1_idx[j]]    
P0 = np.zeros((len(np.where(esti_c==0)[0]),len(np.where(esti_c==0)[0])))
P1 = np.zeros((len(np.where(esti_c==1)[0]),len(np.where(esti_c==1)[0])))            

for i in range(len(P0)):
    for j in range(len(P0)):
        P0[i,j] = B0[j,i]
for i in range(len(P1)):
    for j in range(len(P1)):
        P1[i,j] = B1[j,i]      
     
d_0 = []
for i in range(len(P0)):
    d_0.append(np.sum(P0[i,:]))
d_1 = []
for i in range(len(P1)):
    d_1.append(np.sum(P1[i,:]))
d_max_0 = max(d_0)
d_max_1 = max(d_1)

PT0 = np.zeros((len(np.where(esti_c==0)[0]),len(np.where(esti_c==0)[0])))
PT1 = np.zeros((len(np.where(esti_c==1)[0]),len(np.where(esti_c==1)[0])))   

for i in range(len(PT0)):
    for j in range(len(PT0)):
        if i==j:
            PT0[i,j] = 1-(np.sum(P0[i,:])/d_max_0)
        else:
            PT0[i,j] = P0[i,j]/d_max_0
for i in range(len(PT1)):
    for j in range(len(PT1)):
        if i==j:
            PT1[i,j] = 1-(np.sum(P1[i,:])/d_max_1)
        else:
            PT1[i,j] = P1[i,j]/d_max_1

def sprk(P): 
    P_trans = P.T-np.identity(len(P))
    sum_p = np.ones((len(P_trans),1)).T
    P_sol = np.vstack((P_trans,sum_p))
    b_sol = np.vstack((np.zeros((len(P_trans),1)),np.ones((1,1))))
    pi_stationary = np.linalg.lstsq(P_sol, b_sol,rcond=None)[0]   #ranking
    pi_stationary = np.reshape(pi_stationary,(len(P_trans),))
    return pi_stationary
#%%
esti_s0 = []
esti_s1 = []
for i in range(len(np.where(esti_c==0)[0])):
    esti_s0.append([true_s[np.where(esti_c==0)[0][i]],np.where(esti_c==0)[0][i]])
for i in range(len(np.where(esti_c==1)[0])):
    esti_s1.append([true_s[np.where(esti_c==1)[0][i]],np.where(esti_c==1)[0][i]])  
esti_s0 = np.array(esti_s0)
esti_s1 = np.array(esti_s1)
def rkacc(esti,true):
    true_rank = np.argsort(-true[:,0])
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
    for i in range(len(true[:,0])):
        sp_count=sp_count+np.abs(np.where(true_rank==i)[0][0]-np.where(estimate_rk==i)[0][0])
    return r_acc,sp_count 
r_acc0,r_sps0 = rkacc(sprk(PT0),esti_s0)
r_acc1,r_sps1 = rkacc(sprk(PT1),esti_s1)

#%%
P_re_1 = np.column_stack((sprk(PT0),clu_0_idx))
P_re_2 = np.column_stack((sprk(PT1),clu_1_idx))
P_re_all = np.row_stack((P_re_1,P_re_2))
df = pd.DataFrame(data=P_re_all)
P_re_all = df.sort_values(df.columns[1]).values
#%%
esti_s0_g = []
esti_s1_g = []
true_s0_g = []
true_s1_g = []
for i in range(len(np.where(true_c==0)[0])):
    true_s0_g.append(true_s[np.where(true_c==0)[0][i]])
    esti_s0_g.append(P_re_all[np.where(true_c==0)[0][i]])
for i in range(len(np.where(true_c==1)[0])):
    true_s1_g.append(true_s[np.where(true_c==1)[0][i]])   
    esti_s1_g.append(P_re_all[np.where(true_c==1)[0][i]])
esti_s0_g = np.array(esti_s0_g)
esti_s1_g = np.array(esti_s1_g)
true_s0_g = np.array(true_s0_g)
true_s1_g = np.array(true_s1_g)
r_g_acc0,r_g_sps0 = rkacc(true_s0_g,esti_s0_g)
r_g_acc1,r_g_sps1 = rkacc(true_s1_g,esti_s1_g)


WMW_g_avg = (r_g_acc0+r_g_acc1)/2
SP_g_avg = (r_g_sps0+r_g_sps1)/2
