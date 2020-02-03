
import numpy as np
#================================Initialize===================================
worker_num = 30
item_num = 30
observation = np.load("syn w from x_missing40.npy")
tran_ob = np.zeros((worker_num,item_num,item_num))-1
for k in range(worker_num):
    for i in range(item_num):
        for j in range(i+1,item_num):
            if observation[k,i,j]!=-1:
                if observation[k,i,j]==1:
                    tran_ob[k,i,j] = 1
                    tran_ob[k,j,i] = 0
                else:
                    tran_ob[k,i,j] = 0
                    tran_ob[k,j,i] = 1
            else:
                continue


# load true score (sunthetic dataset)
true_s = np.load("syn true_s w from x.npy")
#load true class (synthetic dataset)
true_c = np.load("syn true_c w from x.npy")
true_c = np.argmax(true_c,axis=1)
#==============================================================================

eta_worker = np.zeros(worker_num) 
l_ind = list(range(item_num))
l_score = list(true_s)
d = list(zip(l_score,l_ind))
d.sort(reverse = True) #降序
l_score[:],l_ind[:] = zip(*d)
consensus_r_old = np.array(l_ind)
consensus_r_new = np.zeros(item_num)
total_wpair = {}
for k in range(worker_num):
    total_wpair[k] = np.where(observation[k,:,:]!=-1)[0].shape[0]


def worker_qua(consensus_r):
    for k in range(worker_num):
        same_pair = 0

        for i in range(item_num):
            for j in range(i+1,item_num):
                if np.where(consensus_r==i)<np.where(consensus_r==j) and observation[k,i,j]==1:
                    same_pair = same_pair + 1
                if np.where(consensus_r==i)>np.where(consensus_r==j) and observation[k,i,j]==0:
                    same_pair = same_pair + 1
                
        eta_worker[k] = same_pair/total_wpair[k]
        
def rank_dis(w_ind,item_x):

    tmp_list = [[1,0]]
    t = 1
    
    for cur_i in range(item_num):
        if cur_i != item_x:

            P_s_t = 0
            item_x_win = np.where(tran_ob[w_ind,item_x]==1)[0].shape[0]
            cur_i_win = np.where(tran_ob[w_ind,cur_i]==1)[0].shape[0]
            if item_x_win == cur_i_win or tran_ob[w_ind,item_x_win,cur_i_win]==-1:
                P_s_t = 0.5
            else:
                if item_x_win > cur_i_win:
                    P_s_t = (item_x_win - cur_i_win)/item_num
                else: 
                    P_s_t = 1 - ((cur_i_win - item_x_win)/item_num)
                    
            P_s_t = eta_worker[w_ind]*P_s_t + (1-eta_worker[w_ind])*(1-P_s_t)    
            tmp_list.append(list(np.convolve(tmp_list[t-1],[P_s_t,(1-P_s_t)])))
            t = t + 1
    
    return tmp_list[-1]                
#==========================ListwiseApproach method============================

rankdis_dic = {}
iter_num = 0


while 1:
    print(iter_num)
#================================Algorithm 1===================================

    worker_qua(consensus_r_old)
    for k in range(worker_num):
        for i in range(item_num):
            tmpind = "w_"+str(k)+"item_"+str(i)
            rankdis_dic[tmpind] = rank_dis(k,i)
#==============================================================================

#==============consruct optimization algorithm for updating \pi================

    RBP_p = 0.95
    tmp_eva_val = np.zeros(item_num) 
    tmpindsort = list(range(item_num))
    for i in range(item_num):     
        for k in range(worker_num):  
            tmp_val = 0
            for r in range(item_num):
                index = "w_"+str(k)+"item_"+str(i)
                tmp_RBPterm2 = rankdis_dic[index][r]
                tmp_val = tmp_val + (RBP_p**(r-1))*tmp_RBPterm2
            tmp_eva_val[i] = tmp_eva_val[i] + (1-RBP_p)*tmp_val   
    
    tmp_record = []

    c = list(zip(list(tmp_eva_val),tmpindsort))
    c.sort(reverse = False) 
    tmp_record[:],tmpindsort[:] = zip(*c)
    
    consensus_new = np.array(tmpindsort)
    
    print(consensus_new)    
    print(consensus_r_old) 
    if (consensus_r_old == consensus_new).all() == True:
        print("yes")
        break
    else:
        consensus_r_old = consensus_new.copy()
        
#==============================================================================
        
#==============consruct optimization algorithm for updating eta================
   
    iter_num = iter_num + 1
#==============================================================================     
 
esti_s0 = []
esti_s1 = []
#esti_s2 = []
#esti_s3 = []
true_s0 = []
true_s1 = []
#true_s2 = []
#true_s3 = []

true_s0.extend(list(np.where(true_c==0)[0]))
true_s1.extend(list(np.where(true_c==1)[0]))
#true_s2.extend(list(np.where(true_c==2)[0]))
#true_s3.extend(list(np.where(true_c==3)[0]))

true_score_0 = []
true_score_1 = []
#true_score_2 = []
#true_score_3 = []

for i in true_s0:
    true_score_0.append(true_s[i])

for i in true_s1:
    true_score_1.append(true_s[i])
    
#for i in true_s2:
#    true_score_2.append(true_s[i])
#    
#for i in true_s3:
#    true_score_3.append(true_s[i])
    

    
rank_deno0 = 0
rank_num0 = 0
for i_ind in true_s0:
    for j_ind in true_s0:
        if i_ind != j_ind:
            
           if true_s[true_s0[true_s0.index(i_ind)]] >= true_s[true_s0[true_s0.index(j_ind)]]:
               rank_deno0 = rank_deno0 + 1
               if np.where(consensus_new==i_ind)[0][0] < np.where(consensus_new==j_ind)[0][0]:
                   rank_num0 = rank_num0 + 1
r_acc0 = rank_num0/rank_deno0

rank_deno1 = 0
rank_num1 = 0
for i_ind in true_s1:
    for j_ind in true_s1:
        if i_ind != j_ind:
            ind1 = true_s1.index(i_ind)
            ind2 = true_s1.index(j_ind)
            if true_s[true_s1[ind1]] >= true_s[true_s1[ind2]]:
                rank_deno1 = rank_deno1 + 1
                if np.where(consensus_new==i_ind)[0][0] < np.where(consensus_new==j_ind)[0][0]:
                    rank_num1 = rank_num1 + 1
r_acc1 = rank_num1/rank_deno1




c2 = list(zip(true_score_0,true_s0))
c2.sort(reverse = True) 
true_score_0[:],true_s0[:] = zip(*c2)
spearcon0 = []

for i in range(item_num):
    if consensus_new[i] in true_s0:
        spearcon0.append(consensus_new[i])        
spear_dist0 = 0
for i in true_s0:
    spear_dist0 = spear_dist0 + np.abs((true_s0.index(i) - spearcon0.index(i)))  
    
c3 = list(zip(true_score_1,true_s1))
c3.sort(reverse = True) 
true_score_1[:],true_s1[:] = zip(*c3)
spearcon1 = []

for i in range(item_num):
    if consensus_new[i] in true_s1:
        spearcon1.append(consensus_new[i])    
spear_dist1 = 0
for i in true_s1:
    spear_dist1 = spear_dist1 + np.abs((true_s1.index(i) - spearcon1.index(i)))  
    
#c4 = list(zip(true_score_2,true_s2))
#c4.sort(reverse = True) #降序
#true_score_2[:],true_s2[:] = zip(*c4)
#spearcon2 = []
#
#for i in range(item_num):
#    if consensus_new[i] in true_s2:
#        spearcon2.append(consensus_new[i])    
#spear_dist2 = 0
#for i in true_s2:
#    spear_dist2 = spear_dist2 + np.abs((true_s2.index(i) - spearcon2.index(i)))
#
#c5 = list(zip(true_score_3,true_s3))
#c5.sort(reverse = True) #降序
#true_score_3[:],true_s3[:] = zip(*c5)
#spearcon3 = []
#
#for i in range(item_num):
#    if consensus_new[i] in true_s3:
#        spearcon3.append(consensus_new[i])    
#spear_dist3 = 0
#for i in true_s3:
#    spear_dist3 = spear_dist3 + np.abs((true_s3.index(i) - spearcon3.index(i)))



























#==============================================================================
