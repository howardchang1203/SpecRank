# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:01:46 2019

@author: yang9
"""
#=========================limited-memory BFGS algo=============================
import scipy.optimize as opt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import time
from numba import jit


timestart = time.time()
worker_num = 30
item_num = 30
data = np.load("D:\\AISTATS\\Data\\syn\\syn w from x missing80_cover previous miss data.npy")
true_c = np.load("D:\\AISTATS\\Data\\syn\\syn true_c w from x.npy")[:,0]
true_s = np.load("D:\\AISTATS\\Data\\syn\\syn true_s w from x.npy")



re_coe = 1
init_s0 = 1
init_s = np.array(list(range(item_num)))
#points = []
init_eta = np.ones(worker_num)
epsilon = 0.00001
#%%

def re_term(lamb,s0,s):
    to_re = 0
    for i in range(item_num):
        to_re = to_re + (np.log((np.exp(s0)/(np.exp(s0)+np.exp(s[i])))) + np.log((np.exp(s[i])/(np.exp(s0)+np.exp(s[i])))))
    return lamb*to_re

def obj_fun_s(s):
    loss = 0
    regularized_val = re_term(re_coe,init_s0,s)
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):
               if data[k,i,j] != -1: 
                   if data[k,i,j]==1:
                       loss = loss + np.log(init_eta[k]*(np.exp(s[i])/(np.exp(s[i])+np.exp(s[j])))+(1-init_eta[k])*(np.exp(s[j])/(np.exp(s[i])+np.exp(s[j]))))
                   else:
                       loss = loss + np.log(init_eta[k]*(np.exp(s[j])/(np.exp(s[i])+np.exp(s[j])))+(1-init_eta[k])*(np.exp(s[i])/(np.exp(s[i])+np.exp(s[j]))))
    return -1*(loss+regularized_val)

def obj_fun_eta(eta):
    loss = 0
    regularized_val = re_term(re_coe,init_s0,init_s)
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):
                if data[k,i,j] != -1:
                    if data[k,i,j]==1:
                        loss = loss + np.log(eta[k]*(np.exp(init_s[i])/(np.exp(init_s[i])+np.exp(init_s[j])))+(1-eta[k])*(np.exp(init_s[j])/(np.exp(init_s[i])+np.exp(init_s[j]))))   
                    else:
                        loss = loss + np.log(eta[k]*(np.exp(init_s[j])/(np.exp(init_s[i])+np.exp(init_s[j])))+(1-eta[k])*(np.exp(init_s[i])/(np.exp(init_s[i])+np.exp(init_s[j]))))
    return -1*(loss+regularized_val)

def obj_fun(s,eta):
    loss = 0
    regularized_val = re_term(re_coe,init_s0,s)
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):
                if data[k,i,j]==1:
                    loss = loss + np.log(eta[k]*(np.exp(s[i])/(np.exp(s[i])+np.exp(s[j])))+(1-eta[k])*(np.exp(s[j])/(np.exp(s[i])+np.exp(s[j]))))
                else:
                    loss = loss + np.log(eta[k]*(np.exp(s[j])/(np.exp(s[i])+np.exp(s[j])))+(1-eta[k])*(np.exp(s[i])/(np.exp(s[i])+np.exp(s[j]))))
    return -1*(loss+regularized_val)
#%%

iter_num = 0
obj = []
lb = np.zeros(worker_num)
ub = np.ones(worker_num)
eta_bound = Bounds(lb, ub, keep_feasible=False)
obj.append(obj_fun(init_s,init_eta))
while 1:
    print(iter_num) 
    tmp_s = init_s.copy()
    tmp_eta = init_eta.copy()           
    #init_s = opt.fmin_l_bfgs_b(obj_fun, init_s)
    init_s = minimize(obj_fun_s, init_s, method='L-BFGS-B').x
    init_eta = minimize(obj_fun_eta, init_eta,bounds=eta_bound, method='L-BFGS-B').x
    #print(init_s)
    print(obj_fun(init_s,init_eta))
    obj.append(obj_fun(init_s,init_eta))
    
    if np.abs((obj_fun(init_s,init_eta) - obj_fun(tmp_s,tmp_eta))/obj_fun(tmp_s,tmp_eta)) >= epsilon:
        iter_num = iter_num + 1
        continue
    else:
        init_s = tmp_s.copy()
        init_eta = tmp_eta.copy()
        break
#    return init_s,init_eta


#init_s,init_eta = main(init_eta,init_s)
timeend = time.time()
print('Crowd-BT runs:', timeend - timestart)
#%%


esti_s0 = []
esti_s1 = []
true_s0 = []
true_s1 = []
for i in range(len(np.where(true_c==0)[0])):
    true_s0.append(true_s[np.where(true_c==0)[0][i]])
    esti_s0.append(init_s[np.where(true_c==0)[0][i]])
for i in range(len(np.where(true_c==1)[0])):
    true_s1.append(true_s[np.where(true_c==1)[0][i]])   
    esti_s1.append(init_s[np.where(true_c==1)[0][i]])
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
    r_acc = rank_num/rank_deno    
    sp_count=0
    for i in range(len(true)):
        sp_count=sp_count+np.abs(np.where(true_rank==i)[0][0]-np.where(estimate_rk==i)[0][0])
    return r_acc,sp_count
r_acc0,r_sps0 = rkacc(esti_s0,true_s0)
r_acc1,r_sps1 = rkacc(esti_s1,true_s1)
#r_acc,r_sps = rkacc(init_s,true_s) 

avg_WMW = (r_acc1+r_acc0)/2
avg_SP = (r_sps0+r_sps1)/2

#%% con/jou missing
#esti_s_all = init_s
#esti_s0 = []
#esti_s1 = []
#esti_s2 = []
#esti_s3 = []
#true_s0 = []
#true_s1 = []
#true_s2 = []
#true_s3 = []
#for i in range(len(np.where(true_c==0)[0])):
#    true_s0.append(true_s[np.where(true_c==0)[0][i]])
#    esti_s0.append(esti_s_all[np.where(true_c==0)[0][i]])
#for i in range(len(np.where(true_c==1)[0])):
#    true_s1.append(true_s[np.where(true_c==1)[0][i]])   
#    esti_s1.append(esti_s_all[np.where(true_c==1)[0][i]])
#for i in range(len(np.where(true_c==2)[0])):
#    true_s2.append(true_s[np.where(true_c==2)[0][i]])   
#    esti_s2.append(esti_s_all[np.where(true_c==2)[0][i]])
#for i in range(len(np.where(true_c==3)[0])):
#    true_s3.append(true_s[np.where(true_c==3)[0][i]])   
#    esti_s3.append(esti_s_all[np.where(true_c==3)[0][i]])
#esti_s0 = np.array(esti_s0)
#esti_s1 = np.array(esti_s1)
#esti_s2 = np.array(esti_s2)
#esti_s3 = np.array(esti_s3)
#true_s0 = np.array(true_s0)
#true_s1 = np.array(true_s1)
#true_s2 = np.array(true_s2)
#true_s3 = np.array(true_s3)
#
#def rkacc(esti,true):
#    true_rank = np.argsort(-true)
#    estimate_rk = np.argsort(-esti)
#    rank_deno = 0
#    rank_num = 0
#    for i_ind in range(len(true_rank)):
#           for j_ind in range(len(true_rank)):
#               if i_ind != j_ind: 
#                   if np.where(true_rank==i_ind)[0][0] < np.where(true_rank==j_ind)[0][0]:
#                       rank_deno = rank_deno + 1
#                       if np.where(estimate_rk==i_ind)[0][0] < np.where(estimate_rk==j_ind)[0][0]:
#                           rank_num = rank_num + 1
#    if rank_deno == 0:
#        rank_deno=-1
#    r_acc = rank_num/rank_deno    
#    r_sps = np.sum(np.abs(true_rank-estimate_rk))
#    return r_acc,r_sps     
#r_acc0,r_sps0 = rkacc(esti_s0,true_s0)
#r_acc1,r_sps1 = rkacc(esti_s1,true_s1)
#r_acc2,r_sps2 = rkacc(esti_s2,true_s2)
#r_acc3,r_sps3 = rkacc(esti_s3,true_s3)
#
##%%
#avg_WMW = (r_acc1+r_acc0+r_acc2+r_acc3)/4
#avg_SP = (r_sps0+r_sps1+r_sps2+r_sps3)/4    
    
    
    
    
#%%    
#true_rank_c1 = np.argsort(-true_s[:15])
#estimate_rk_c1 = np.argsort(-init_s[:15])
#rank_dist_c1 = np.sum(np.abs(estimate_rk_c1-true_rank_c1))  
#true_rank_c2 = np.argsort(-true_s[15:])
#estimate_rk_c2 = np.argsort(-init_s[15:])    
#rank_dist_c2 = np.sum(np.abs(estimate_rk_c2-true_rank_c2))    
    
    
    
#%%
#x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
#bounds = Bounds([0, -0.5], [1.0, 2.0])
##res = minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True},bounds=bounds)
#def obj_fun_eta(eta, s):
#    #計算loss
#    loss = 0
#    regularized_val = re_term(re_coe,init_s0,s)
#    for k in range(worker_num):
#        for i in range(item_num):
#            for j in range(i+1,item_num):
#                if observation[k,i,j] != -1:
#                    if observation[k,i,j]==1:
#                        loss = loss + np.log(eta[k]*((np.exp(s[i]))/(np.exp(s[i])+np.exp(s[j])))+(1-eta[k])*((np.exp(s[j]))/(np.exp(s[i])+np.exp(s[j])))) + regularized_val 
#                    else:
#                        loss = loss + np.log(eta[k]*((np.exp(s[j]))/(np.exp(s[i])+np.exp(s[j])))+(1-eta[k])*((np.exp(s[i]))/(np.exp(s[i])+np.exp(s[j])))) + regularized_val 
#                    continue          
#    points.append(loss)
#    return loss
#from scipy.optimize import SR1
#res = minimize(obj_fun_eta, init_eta, method='trust-constr',  jac="2-point", hess=SR1(),options={'verbose': 1}, bounds=bounds,args=(result[0]))


s_listwise = np.argsort(-init_s)
