
import random
#import math
import numpy as np
from sklearn.cluster import KMeans
#worker num
worker_num = 30
#item number
item_num = 30
#dimension for the vectors
dim = 2

#================================Initialize===================================
pairnum = 30
#cluster num
cluster = 2
#regularization val
#reg_weight = 0.1
#virtual node val
#s0 = 1
observation = np.load("syn observation w from x.npy")

gamma_1 = 0.0001
gamma_2 = 0.00001
gamma_3 = 0.00001

'''
Init = np.load("reform initial 5A.npz")
train_w = Init["init_w"]
train_x = Init["init_x"]
train_beta = Init["init_beta"]
train_P = Init["init_P"]
'''

train_w = np.zeros((cluster,dim))
train_x = np.random.random((item_num,dim))
kmeans = KMeans(n_clusters=cluster, random_state=0).fit(train_x)

for pre_c in range(cluster):
    preind = np.where(kmeans.labels_== pre_c)[0]
    train_w[pre_c] = (np.sum(train_x[preind],axis=0)/train_x[preind].shape[0]).copy()
    train_w[pre_c] = train_w[pre_c]/(np.dot(train_w[pre_c],train_w[pre_c].T)**(1/2))
#normvec = (np.dot(train_x,train_x.T).diagonal())**(1/2)
#for i in range(item_num):
    #train_x[i] = train_x[i]/normvec[i]
    
train_beta = np.random.random((worker_num,dim+1))
train_P = (np.random.multinomial(100, [1/2.]*2, size=1)/100).reshape(cluster)

true_s = np.load("syn true_s w from x.npy")
true_c = np.load("syn true_c w from x.npy")


initial_w = train_w.copy()
initial_x = train_x.copy()
initial_beta = train_beta.copy()
initial_P = train_P.copy()
#==============================================================================

#np.savez("specrank_regularized reformed on com_syn initpara_dim4_3", init_w=initial_w, init_x=initial_x,init_beta=initial_beta,init_P=initial_P)

def likelihood_ykij_1(w_c1,w_c2,x_i,x_j,beta_k):
    
    #w = np.c_[np.ones((cluster,1)),w_c1+w_c2]
    w = w_c1 + w_c2
    ans = (1/(1+np.exp(-1*(np.dot(beta_k[1:],w.T)+beta_k[0]))))*((np.exp(np.dot(w_c1,x_i.T)))/(np.exp(np.dot(w_c1,x_i.T))+np.exp(np.dot(w_c2,x_j.T))))+(1-(1/(1+np.exp(-1*(np.dot(beta_k[1:],w.T)+beta_k[0])))))*((np.exp(np.dot(w_c2,x_j.T)))/(np.exp(np.dot(w_c1,x_i.T))+np.exp(np.dot(w_c2,x_j.T))))
    return ans

def likelihood_ykij_0(w_c1,w_c2,x_i,x_j,beta_k):
    #w = np.c_[np.ones((cluster,1)),w_c1+w_c2]
    w = w_c1 + w_c2
    ans1 = (1/(1+np.exp(-1*(np.dot(beta_k[1:],w.T)+beta_k[0]))))*((np.exp(np.dot(w_c2,x_j.T)))/(np.exp(np.dot(w_c1,x_i.T))+np.exp(np.dot(w_c2,x_j.T))))+(1-(1/(1+np.exp(-1*(np.dot(beta_k[1:],w.T)+beta_k[0])))))*((np.exp(np.dot(w_c1,x_i.T)))/(np.exp(np.dot(w_c1,x_i.T))+np.exp(np.dot(w_c2,x_j.T))))
    return ans1


def exp_denom(cluster,y_kij,itemi,P,W,X,Beta,fix_c):
    denom = 0
    
    for avec in range(cluster):
        denom = denom + np.exp(exp_jointdis(cluster,y_kij,itemi,P,W,X,Beta,avec)-exp_jointdis(cluster,y_kij,itemi,P,W,X,Beta,fix_c)) 
        
    return denom


def exp_jointdis(cluster,y_kij,itemi,P,W,X,Beta,item_c_ind):
    total_exp = 0
    
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):
               
                if y_kij[k,i,j]==-1:
                    continue
                else:
                    if i==itemi:
                        for c1 in range(cluster):
                           
                            if c1==item_c_ind:
                                for c2 in range(cluster):
                                    if y_kij[k,i,j]==1:
                                        total_exp = total_exp + (1*P[c2]*np.log(likelihood_ykij_1(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:]))) 
                                    else:
                                        total_exp = total_exp + (1*P[c2]*np.log(likelihood_ykij_0(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:])))
                            else:
                                for c2 in range(cluster):
                                    if y_kij[k,i,j]==1:
                                        total_exp = total_exp + (0*P[c2]*np.log(likelihood_ykij_1(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:]))) 
                                    else:
                                        total_exp = total_exp + (0*P[c2]*np.log(likelihood_ykij_0(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:])))
                    elif j==itemi:
                        for c1 in range(cluster):
                            for c2 in range(cluster):
                                if c2==item_c_ind:
                                    
                                    if y_kij[k,i,j]==1:
                                        total_exp = total_exp + (P[c1]*1*np.log(likelihood_ykij_1(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:]))) 
                                    else:
                                        total_exp = total_exp + (P[c1]*1*np.log(likelihood_ykij_0(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:])))
                                else:
                                    if y_kij[k,i,j]==1:
                                        total_exp = total_exp + (P[c1]*0*np.log(likelihood_ykij_1(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:]))) 
                                    else:
                                        total_exp = total_exp + (P[c1]*0*np.log(likelihood_ykij_0(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:])))
                    else:
                        for c1 in range(cluster):
                            for c2 in range(cluster):
                                if y_kij[k,i,j]==1:
                                    total_exp = total_exp + (P[c1]*P[c2]*np.log(likelihood_ykij_1(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:]))) 
                                else:
                                    total_exp = total_exp + (P[c1]*P[c2]*np.log(likelihood_ykij_0(W[c1,:],W[c2,:],X[i,:],X[j,:],Beta[k,:])))

    for i in range(item_num):
        
        if i == itemi:
            total_exp = total_exp + np.log(P[item_c_ind])
        else:
            for c in range(cluster):
            
                total_exp = total_exp + (P[c]*np.log(P[c]))
   
    return total_exp
               
def Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,update_w,update_x,update_beta,update_P,exp_a):
    Expectation_val = 0  
    print("Q")
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):
                #print("k=" + str(k) + ", i=" + str(i) + ", j=" + str(j))
                k_answer = observation[k,i,j]
                if k_answer != -1:
                    if k_answer==1:
                        for c1 in range(cluster):
                            for c2 in range(cluster):
                                Expectation_val = Expectation_val + exp_a[i][c1]*exp_a[j][c2]*np.log(likelihood_ykij_1(update_w[c1,:],update_w[c2,:],update_x[i,:],update_x[j,:],update_beta[k,:]))
                    else:
                        for c1 in range(cluster):
                            for c2 in range(cluster):
                                Expectation_val = Expectation_val + exp_a[i][c1]*exp_a[j][c2]*np.log(likelihood_ykij_0(update_w[c1,:],update_w[c2,:],update_x[i,:],update_x[j,:],update_beta[k,:]))
                else:
                    continue
                
    for i in range(item_num):
        for c in range(cluster):
            Expectation_val = Expectation_val + exp_a[i][c]*np.log(update_P[c])
    '''        
    tmpreg = 0
    for c in range(cluster):
        for i in range(item_num):
            tmpreg = tmpreg + np.log((np.exp(s0)/(np.exp(s0) + np.exp(np.dot(update_w[c],update_x[i].T))))) + np.log((np.exp(np.dot(update_w[c],update_x[i].T))/(np.exp(s0) + np.exp(np.dot(update_w[c],update_x[i].T)))))

    tmpreg = reg_weight*tmpreg     
    '''
    tmpreg = 0
    tmpreg = tmpreg + gamma_1*np.sum(np.dot(update_w,update_w.T).diagonal()) + gamma_2*np.sum(np.dot(update_x,update_x.T).diagonal()) + gamma_3*np.sum(np.dot(update_beta,update_beta.T).diagonal())
    
    return Expectation_val - tmpreg
            
#------------------------------------------------------------------------------

#--------------------------------sigmoid func----------------------------------
                      
def sigmoida(wc1,wc2,beta):
    w = wc1+wc2
    expval = np.dot(w,beta[1:].T) + beta[0]
    sigmoidv1 = (1/(1+np.exp(-1*expval)))
    return sigmoidv1

def sigmoidb(wc1,wc2,xi,xj):
    expval = np.dot(wc1,xi.T) - np.dot(wc2,xj.T)
    sigmoidv2 = (1/(1+np.exp(-1*expval)))
    return sigmoidv2

def sigmoidh(wg,beta):
    expval = 2*np.dot(wg,beta[1:].T) + beta[0]
    sigmoidv4 = (1/(1+np.exp(-1*expval)))
    return sigmoidv4

def sigmoidl(wg,xi,xj):
    x = xi - xj
    expval = np.dot(wg,x.T) 
    sigmoidv5 = (1/(1+np.exp(-1*expval)))
    return sigmoidv5
#---------------------------------u gradient-----------------------------------             
def gr_w(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,update_wind,update_w,exp_a):
    
    diff_term = 0
    for k in range(worker_num):
        for i in range(item_num):
            for j in range(i+1,item_num):            
                if observation[k,i,j] != -1:
                    if observation[k,i,j] == 1:
                        for c2 in range(cluster):
                            if c2 != update_wind:
                                frac_likelihood = 1/likelihood_ykij_1(update_w[update_wind],update_w[c2],train_x[i],train_x[j],train_beta[k])
                                tmpterm = (2*sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j])-1)*sigmoida(update_w[update_wind],update_w[c2],train_beta[k])*(1-sigmoida(update_w[update_wind],update_w[c2],train_beta[k]))*train_beta[k][1:]+(2*sigmoida(update_w[update_wind],update_w[c2],train_beta[k])-1)*sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j])*(1-sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j]))*train_x[i]
                                diff_term = diff_term + exp_a[i,update_wind]*exp_a[j,c2]*frac_likelihood*tmpterm
                
                    else:
                        for c2 in range(cluster):
                            if c2 != update_wind:
                                frac_likelihood = 1/likelihood_ykij_0(update_w[update_wind],update_w[c2],train_x[i],train_x[j],train_beta[k])
                                tmpterm = (1-2*sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j]))*sigmoida(update_w[update_wind],update_w[c2],train_beta[k])*(1-sigmoida(update_w[update_wind],update_w[c2],train_beta[k]))*train_beta[k][1:]+(1-2*sigmoida(update_w[update_wind],update_w[c2],train_beta[k]))*sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j])*(1-sigmoidb(update_w[update_wind],update_w[c2],train_x[i],train_x[j]))*train_x[i]
                                diff_term = diff_term + exp_a[i,update_wind]*exp_a[j,c2]*frac_likelihood*tmpterm
                    
                    if observation[k,i,j] == 1:
                        for c1 in range(cluster):
                            if c1 != update_wind:
                                frac_likelihood = 1/likelihood_ykij_1(update_w[c1],update_w[update_wind],train_x[i],train_x[j],train_beta[k])       
                                tmpterm = (2*sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j])-1)*sigmoida(update_w[c1],update_w[update_wind],train_beta[k])*(1-sigmoida(update_w[c1],update_w[update_wind],train_beta[k]))*train_beta[k][1:]+(1-2*sigmoida(update_w[c1],update_w[update_wind],train_beta[k]))*sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j])*(1-sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j]))*train_x[j]
                                diff_term = diff_term + exp_a[i,c1]*exp_a[j,update_wind]*frac_likelihood*tmpterm
                
                    else:
                        for c1 in range(cluster):
                            if c1 != update_wind:
                                frac_likelihood = 1/likelihood_ykij_0(update_w[c1],update_w[update_wind],train_x[i],train_x[j],train_beta[k])
                                tmpterm = (1-2*sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j]))*sigmoida(update_w[c1],update_w[update_wind],train_beta[k])*(1-sigmoida(update_w[c1],update_w[update_wind],train_beta[k]))*train_beta[k][1:]+(2*sigmoida(update_w[c1],update_w[update_wind],train_beta[k])-1)*sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j])*(1-sigmoidb(update_w[c1],update_w[update_wind],train_x[i],train_x[j]))*train_x[j]
                                diff_term = diff_term + exp_a[i,c1]*exp_a[j,update_wind]*frac_likelihood*tmpterm
                
                    if observation[k,i,j] == 1:
                        frac_likelihood = 1/likelihood_ykij_1(update_w[update_wind],update_w[update_wind],train_x[i],train_x[j],train_beta[k])           
                        tmpterm = (2*sigmoidl(update_w[update_wind],train_x[i],train_x[j])-1)*sigmoidh(update_w[update_wind],train_beta[k])*(1-sigmoidh(update_w[update_wind],train_beta[k]))*2*train_beta[k][1:]+(2*sigmoidh(update_w[update_wind],train_beta[k])-1)*sigmoidl(update_w[update_wind],train_x[i],train_x[j])*(1-sigmoidl(update_w[update_wind],train_x[i],train_x[j]))*(train_x[i]-train_x[j])
                        diff_term = diff_term + exp_a[i,update_wind]*exp_a[j,update_wind]*frac_likelihood*tmpterm
                    else:
                        frac_likelihood = 1/likelihood_ykij_0(update_w[update_wind],update_w[update_wind],train_x[i],train_x[j],train_beta[k])
                        tmpterm = (1-2*sigmoidl(update_w[update_wind],train_x[i],train_x[j]))*sigmoidh(update_w[update_wind],train_beta[k])*(1-sigmoidh(update_w[update_wind],train_beta[k]))*2*train_beta[k][1:]+(1-2*sigmoidh(update_w[update_wind],train_beta[k]))*sigmoidl(update_w[update_wind],train_x[i],train_x[j])*(1-sigmoidl(update_w[update_wind],train_x[i],train_x[j]))*(train_x[i]-train_x[j])
                        diff_term = diff_term + exp_a[i,update_wind]*exp_a[j,update_wind]*frac_likelihood*tmpterm
                
                else:
                    continue
                
    tmpregw = 0
    tmpregw = tmpregw + 2*gamma_1*update_w[update_wind]
    
    return diff_term - tmpregw

#-----------------------------------v gradient---------------------------------
def gr_x(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,update_xind,update_w,update_x,exp_a):    
    diff_term = 0    
    for k in range(worker_num):
        for j in range(update_xind+1,item_num):
            if observation[k,update_xind,j] != -1:
                if observation[k,update_xind,j] == 1:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_1(update_w[c1],update_w[c2],update_x[update_xind],update_x[j],train_beta[k])
                            tmpterm = (2*sigmoida(update_w[c1],update_w[c2],train_beta[k])-1)*sigmoidb(update_w[c1],update_w[c2],update_x[update_xind],update_x[j])*(1-sigmoidb(update_w[c1],update_w[c2],update_x[update_xind],update_x[j]))*update_w[c1]
                            diff_term = diff_term + exp_a[update_xind,c1]*exp_a[j,c2]*frac_likelihood*tmpterm
                else:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_0(update_w[c1],update_w[c2],update_x[update_xind],update_x[j],train_beta[k])
                            tmpterm = (1-2*sigmoida(update_w[c1],update_w[c2],train_beta[k]))*sigmoidb(update_w[c1],update_w[c2],update_x[update_xind],update_x[j])*(1-sigmoidb(update_w[c1],update_w[c2],update_x[update_xind],update_x[j]))*update_w[c1]
                            diff_term = diff_term + exp_a[update_xind,c1]*exp_a[j,c2]*frac_likelihood*tmpterm
          
            else:
                continue
                     
    for k in range(worker_num):
        for i in range(update_xind):
            if observation[k,i,update_xind] != -1:
                if observation[k,i,update_xind] == 1:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_1(update_w[c1],update_w[c2],update_x[i],update_x[update_xind],train_beta[k])
                            tmpterm = (1-2*sigmoida(update_w[c1],update_w[c2],train_beta[k]))*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[update_xind])*(1-sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[update_xind]))*update_w[c2]
                            diff_term = diff_term + exp_a[i,c1]*exp_a[update_xind,c2]*frac_likelihood*tmpterm
                else:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_0(update_w[c1],update_w[c2],update_x[i],update_x[update_xind],train_beta[k])
                            tmpterm = (2*sigmoida(update_w[c1],update_w[c2],train_beta[k])-1)*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[update_xind])*(1-sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[update_xind]))*update_w[c2]
                            diff_term = diff_term + exp_a[i,c1]*exp_a[update_xind,c2]*frac_likelihood*tmpterm            
            else:
                continue

    tmpregx = 0
    tmpregx = tmpregx + 2*gamma_2*update_x[update_xind]     
    
    return diff_term - tmpregx
#------------------------------------------------------------------------------
#-------------------------------beta_k0 gradient------------------------------  
def gr_beta_ko(item_num,cluster,observation,train_w,train_x,train_beta,train_P,update_kind,update_w,update_x,exp_a):
    diff_term = 0 
    for i in range(item_num):
        for j in range(i+1,item_num):
            if observation[update_kind,i,j] != -1:
                if observation[update_kind,i,j] == 1:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_1(update_w[c1],update_w[c2],update_x[i],update_x[j],train_beta[update_kind])
                            tmpterm = (2*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[j])-1)*sigmoida(update_w[c1],update_w[c2],train_beta[update_kind])*(1-sigmoida(update_w[c1],update_w[c2],train_beta[update_kind]))
                            diff_term = diff_term + exp_a[i,c1]*exp_a[j,c2]*frac_likelihood*tmpterm
                else:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_0(update_w[c1],update_w[c2],update_x[i],update_x[j],train_beta[update_kind])
                            tmpterm = (1-2*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[j]))*sigmoida(update_w[c1],update_w[c2],train_beta[update_kind])*(1-sigmoida(update_w[c1],update_w[c2],train_beta[update_kind]))
                            diff_term = diff_term + exp_a[i,c1]*exp_a[j,c2]*frac_likelihood*tmpterm           
            else:
                continue
            
    tmpregk0 = 0
    tmpregk0 = tmpregk0 + 2*gamma_3*train_beta[update_kind,0]      
        
    return diff_term - tmpregk0
#------------------------------------------------------------------------------
def gr_beta_k1(item_num,cluster,observation,train_w,train_x,train_beta,train_P,update_kind,update_w,update_x,update_beta,exp_a):
    diff_term = 0 
    for i in range(item_num):
        for j in range(i+1,item_num):
            if observation[update_kind,i,j] != -1:
                if observation[update_kind,i,j] == 1:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_1(update_w[c1],update_w[c2],update_x[i],update_x[j],update_beta[update_kind])
                            tmpterm = (2*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[j])-1)*sigmoida(update_w[c1],update_w[c2],update_beta[update_kind])*(1-sigmoida(update_w[c1],update_w[c2],update_beta[update_kind]))*(update_w[c1]+update_w[c2])
                            diff_term = diff_term + exp_a[i,c1]*exp_a[j,c2]*frac_likelihood*tmpterm
                else:
                    for c1 in range(cluster):
                        for c2 in range(cluster):
                            frac_likelihood = 1/likelihood_ykij_0(update_w[c1],update_w[c2],update_x[i],update_x[j],update_beta[update_kind])
                            tmpterm = (1-2*sigmoidb(update_w[c1],update_w[c2],update_x[i],update_x[j]))*sigmoida(update_w[c1],update_w[c2],update_beta[update_kind])*(1-sigmoida(update_w[c1],update_w[c2],update_beta[update_kind]))*(update_w[c1]+update_w[c2])
                            diff_term = diff_term + exp_a[i,c1]*exp_a[j,c2]*frac_likelihood*tmpterm
            else:
                continue
    
    tmpregk1 = 0
    tmpregk1 = tmpregk1 + 2*gamma_3*train_beta[update_kind][1:]
     
    return diff_term - tmpregk1

w_learning_rate = 0.005
x_learning_rate = 0.05
beta_k0_learning_rate = 0.05
beta_k1_learning_rate = 0.05

epsilon = 0.000001

acclist = []
rankacc = []
spearman = []
es_ranking = []
tr_c = np.argmax(true_c,axis=1)
toiter = 0
rankacc_esbased = []
spearman_esbased = []
update_expa = np.zeros((item_num,cluster),dtype=float)
exp1 = []
exp2 = []
while 1:       
    
    w_tmp = train_w.copy()
    x_tmp = train_x.copy()
    beta_tmp = train_beta.copy()
    P_tmp = train_P.copy()

    topa_update = 0

    for i in range(item_num):
        for c in range(cluster):
            update_expa[i,c] = 1/exp_denom(cluster,observation,i,train_P,train_w,train_x,train_beta,c)

    for w_ind in range(cluster):
        
        update_count = 0
        Q_corr_w_old = 0
        Q_corr_w_new = 0
        
        while 1:
            if Q_corr_w_old == 0:
                # default
                print("w_update_" + str(w_ind) + ": " + str(update_count))
                Q_corr_w_old = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,train_x,train_beta,train_P,update_expa)              
                gradient_w = gr_w(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_ind,w_tmp,update_expa) 
                tmpupdate = w_tmp[w_ind].copy()
                w_tmp[w_ind] = w_tmp[w_ind] + w_learning_rate*gradient_w
                Q_corr_w_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,train_x,train_beta,train_P,update_expa)
                print("Q_corr_w_old:" + str(Q_corr_w_old) + " ; " + " Q_corr_w_new:" + str(Q_corr_w_new))
                #print("calculate w_update")
                if ((Q_corr_w_new - Q_corr_w_old)/np.abs(Q_corr_w_old)) >= epsilon : 
                    update_count = update_count + 1                    
                    Q_corr_w_old = Q_corr_w_new                  
                    print("w_update_num=" + str(update_count) + "," + "w" + str(w_ind) +"_corr_Q=" + str(Q_corr_w_new) )
                else:    
                    w_tmp[w_ind] =  tmpupdate.copy()
                    break
            else:
                #default
                print("w_update_" + str(w_ind) + ": " + str(update_count))
                gradient_w = gr_w(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_ind,w_tmp,update_expa)  
                tmpupdate = w_tmp[w_ind].copy()
                w_tmp[w_ind] = w_tmp[w_ind] + w_learning_rate*gradient_w
                Q_corr_w_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,train_x,train_beta,train_P,update_expa)
                print("Q_corr_w_old:" + str(Q_corr_w_old) + " ; " + " Q_corr_w_new:" + str(Q_corr_w_new))
                if ((Q_corr_w_new - Q_corr_w_old)/np.abs(Q_corr_w_old)) >= epsilon : 
                    update_count = update_count + 1
                    Q_corr_w_old = Q_corr_w_new
                    print("w_update_num=" + str(update_count) + "," + "w" + str(w_ind) +"_corr_Q=" + str(Q_corr_w_new) )
                else:
                    w_tmp[w_ind] = tmpupdate.copy()
                    break
                
        
#------------------------------------------------------------------------------
#----------------------------------udpate v------------------------------------

    for x_ind in range(item_num):
        update_count = 0
        Q_corr_x_old = 0
        Q_corr_x_new = 0
        #x_tmp_update = train_x[x_ind].copy()
        while 1:
            
            if Q_corr_x_old == 0: 
                print("x_update_" + str(x_ind) + ": " + str(update_count))
                Q_corr_x_old = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,train_beta,train_P,update_expa)
                gradient_x = gr_x(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,x_ind,w_tmp,x_tmp,update_expa)    
                tmpupdate = x_tmp[x_ind].copy()
                x_tmp[x_ind] = x_tmp[x_ind] + x_learning_rate*gradient_x
                Q_corr_x_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,train_beta,train_P,update_expa)
                print("Q_corr_x_old:" + str(Q_corr_x_old) + " ; " + " Q_corr_x_new:" + str(Q_corr_x_new))
                if ((Q_corr_x_new - Q_corr_x_old)/np.abs(Q_corr_x_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_x_old = Q_corr_x_new
                    print("x_update_num=" + str(update_count) + "," + "x" + str(x_ind) +"_corr_w=" + str(Q_corr_x_new) )
                else:
                    x_tmp[x_ind] =  tmpupdate.copy()
                    break
            else:
                print("x_update_" + str(x_ind) + ": " + str(update_count))
                gradient_x = gr_x(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,x_ind,w_tmp,x_tmp,update_expa)     
                tmpupdate = x_tmp[x_ind].copy()
                x_tmp[x_ind] = x_tmp[x_ind] + x_learning_rate*gradient_x
                Q_corr_x_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,train_beta,train_P,update_expa)
                print("Q_corr_x_old:" + str(Q_corr_x_old) + " ; " + " Q_corr_x_new:" + str(Q_corr_x_new))
                if ((Q_corr_x_new - Q_corr_x_old)/np.abs(Q_corr_x_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_x_old = Q_corr_x_new
                    print("x_update_num=" + str(update_count) + "," + "x" + str(x_ind) +"_corr_w=" + str(Q_corr_x_new) )
                else:
                    x_tmp[x_ind] = tmpupdate.copy()
                    break
    
    #beta_k0 update
    for k_ind in range(worker_num):
        update_count = 0
        Q_corr_beta_k0_old = 0
        Q_corr_beta_k0_new = 0
        while 1:
            if Q_corr_beta_k0_old == 0:
                print("beta_k0_update_" + str(k_ind) + ": " + str(update_count))
                Q_corr_beta_k0_old = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                gradient_beta_k0 = gr_beta_ko(item_num,cluster,observation,train_w,train_x,train_beta,train_P,k_ind,w_tmp,x_tmp,update_expa)    
                tmpupdate = beta_tmp[k_ind,0].copy()
                beta_tmp[k_ind,0] = beta_tmp[k_ind,0] + beta_k0_learning_rate*gradient_beta_k0
                Q_corr_beta_k0_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                print("Q_corr_beta_k0_old:" + str(Q_corr_beta_k0_old) + " ; " + " Q_corr_beta_k0_new:" + str(Q_corr_beta_k0_new))
                if ((Q_corr_beta_k0_new - Q_corr_beta_k0_old)/np.abs(Q_corr_beta_k0_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_beta_k0_old = Q_corr_beta_k0_new
                    print("beta_k0_update_num=" + str(update_count) + "," + "beta_k0_" + str(k_ind) +"_corr_w=" + str(Q_corr_beta_k0_new))
                else:
                    beta_tmp[k_ind,0] = tmpupdate.copy()
                    break
            else:
                print("beta_k0_update_" + str(k_ind) + ": " + str(update_count))
                gradient_beta_k0 = gr_beta_ko(item_num,cluster,observation,train_w,train_x,train_beta,train_P,k_ind,w_tmp,x_tmp,update_expa)    
                tmpupdate = beta_tmp[k_ind,0].copy()
                beta_tmp[k_ind,0] = beta_tmp[k_ind,0] + beta_k0_learning_rate*gradient_beta_k0
                Q_corr_beta_k0_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                print("Q_corr_beta_k0_old:" + str(Q_corr_beta_k0_old) + " ; " + " Q_corr_beta_k0_new:" + str(Q_corr_beta_k0_new))
                if ((Q_corr_beta_k0_new - Q_corr_beta_k0_old)/np.abs(Q_corr_beta_k0_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_beta_k0_old = Q_corr_beta_k0_new
                    print("beta_k0_update_num=" + str(update_count) + "," + "beta_k0_" + str(k_ind) +"_corr_w=" + str(Q_corr_beta_k0_new))
                else:
                    beta_tmp[k_ind,0] = tmpupdate.copy()
                    break
                
            
    #beta_k1 update
    for k_ind in range(worker_num):
        update_count = 0
        Q_corr_beta_k1_old = 0
        Q_corr_beta_k1_new = 0
        while 1:
            if Q_corr_beta_k1_old == 0:
                print("beta_k1_update_" + str(k_ind) + ": " + str(update_count))
                Q_corr_beta_k1_old = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                gradient_beta_k1 = gr_beta_k1(item_num,cluster,observation,train_w,train_x,train_beta,train_P,k_ind,w_tmp,x_tmp,beta_tmp,update_expa)    
                tmpupdate = beta_tmp[k_ind][1:].copy()
                beta_tmp[k_ind][1:] = beta_tmp[k_ind][1:] + beta_k1_learning_rate*gradient_beta_k1
                Q_corr_beta_k1_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                print("Q_corr_beta_k1_old:" + str(Q_corr_beta_k1_old) + " ; " + " Q_corr_beta_k1_new:" + str(Q_corr_beta_k1_new))
                if ((Q_corr_beta_k1_new - Q_corr_beta_k1_old)/np.abs(Q_corr_beta_k1_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_beta_k1_old = Q_corr_beta_k1_new
                    print("beta_k1_update_num=" + str(update_count) + "," + "beta_k1_" + str(k_ind) +"_corr_w=" + str(Q_corr_beta_k1_new))
                else:
                    beta_tmp[k_ind][1:] = tmpupdate.copy()
                    break
            else:
                print("beta_k1_update_" + str(k_ind) + ": " + str(update_count))
                gradient_beta_k1 = gr_beta_k1(item_num,cluster,observation,train_w,train_x,train_beta,train_P,k_ind,w_tmp,x_tmp,beta_tmp,update_expa)    
                tmpupdate = beta_tmp[k_ind][1:].copy()
                beta_tmp[k_ind][1:] = beta_tmp[k_ind][1:] + beta_k1_learning_rate*gradient_beta_k1
                Q_corr_beta_k1_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,train_P,update_expa)
                print("Q_corr_beta_k1_old:" + str(Q_corr_beta_k1_old) + " ; " + " Q_corr_beta_k1_new:" + str(Q_corr_beta_k1_new))
                if ((Q_corr_beta_k1_new - Q_corr_beta_k1_old)/np.abs(Q_corr_beta_k1_old)) >= epsilon: 
                    update_count = update_count + 1
                    Q_corr_beta_k1_old = Q_corr_beta_k1_new
                    print("beta_k1_update_num=" + str(update_count) + "," + "beta_k1_" + str(k_ind) +"_corr_w=" + str(Q_corr_beta_k1_new))
                else:
                    beta_tmp[k_ind][1:] = tmpupdate.copy()
                    break

#---------------------------update P----------------------------
    for c in range(cluster):      
        P_tmp[c] = np.sum(update_expa[:,c])/item_num
        print("P_" + str(c))
    


    Q_value_old = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,train_w,train_x,train_beta,train_P,update_expa)
    Q_value_new = Q(worker_num,item_num,cluster,observation,train_w,train_x,train_beta,train_P,w_tmp,x_tmp,beta_tmp,P_tmp,update_expa)
    sa = 0  
    es_c = np.argmax(update_expa,axis=1)
    for i in range(item_num):
        for j in range(i+1,item_num):
            if (tr_c[i]==tr_c[j]) and (es_c[i]==es_c[j]):
                sa = sa + 1
            if (tr_c[i]!=tr_c[j]) and (es_c[i]!=es_c[j]):
                sa = sa + 1
    toacc = sa/435  
    acclist.append(toacc) 
    x_num = int(item_num/cluster)
    label_estimated = np.argmax(update_expa,axis=1)
    for c in range(cluster):
        x_corr_ind = label_estimated[c*x_num:(c+1)*x_num]
        x_corr_w = train_w[x_corr_ind]
        corr_x = train_x[c*x_num:(c+1)*x_num]
        estimated_score = np.dot(x_corr_w,corr_x.T).diagonal()
        estimated_ranking = np.argsort(-estimated_score) + c*x_num
        true_rank = np.argsort(-true_s[c*x_num:(c+1)*x_num]) + c*x_num
        rank_deno = 0
        rank_num = 0
        for i_ind in range(c*x_num,(c+1)*x_num):
            for j_ind in range(c*x_num,(c+1)*x_num):
                if i_ind != j_ind: 
                    if np.where(true_rank==i_ind)[0][0] < np.where(true_rank==j_ind)[0][0]:
                        rank_deno = rank_deno + 1
                        if np.where(estimated_ranking==i_ind)[0][0] < np.where(estimated_ranking==j_ind)[0][0]:
                            rank_num = rank_num + 1
        r_acc = rank_num/rank_deno
        rankacc.append(r_acc)
        
        rank_dist = 0
        for i_ind in range(c*x_num,(c+1)*x_num):
            rank_dist = rank_dist + np.abs(np.where(true_rank==i_ind)[0][0]-np.where(estimated_ranking==i_ind)[0][0])
        spearman.append(rank_dist)
        #default
    print("toiter = " + str(toiter) + " , " + "toacc = " + str(toacc) + ", " + "rankacc_group1 = " + str(rankacc[-2]) + ", " + "rankacc_group2 = " + str(rankacc[-1]) + ", " + "spearman_group1 = " + str(spearman[-2]) + ", " + "spearman_group2 = " + str(spearman[-1]))
    
    if ((Q_value_new - Q_value_old)/np.abs(Q_value_old)) >= 0.000001:
        
        train_w = w_tmp.copy()
        train_x = x_tmp.copy()
        train_beta = beta_tmp.copy()
        train_P = P_tmp.copy()
        toiter = toiter + 1
        #save all the trained parameters
#        np.save("specrank_regularized reformed on missing99_cover previous miss data acclist ", acclist)
#        np.save("specrank_regularized reformed on missing99_cover previous miss data rankacc", rankacc)
#        np.save("specrank_regularized reformed on missing99_cover previous miss data spearman", spearman)
#        #np.save("specrank_regularized reformed on conference rankacc_esbased_dim5_revise_2", rankacc_esbased)
#        #np.save("specrank_regularized reformed on conference spearman_esbased_dim5_revise_2", spearman_esbased)
#        np.savez("specrank_regularized reformed on missing99_cover previous miss data updatepara", update_w=train_w, update_x=train_x,update_beta=train_beta,update_P=train_P)
#        np.savez("specrank_regularized reformed on missing99_cover previous miss data update_expa", update_expa)
        #default
        print("toiter = " + str(toiter) + " , " + "toacc = " + str(toacc) + ", " + "rankacc_group1 = " + str(rankacc[-2]) + ", " + "rankacc_group2 = " + str(rankacc[-1]) + ", " + "spearman_group1 = " + str(spearman[-2]) + ", " + "spearman_group2 = " + str(spearman[-1]))
        #print("toiter = " + str(toiter) + " , " + "toacc = " + str(toacc) + ", " + "rankacc_group1 = " + str(rankacc_esbased[-2]) + ", " + "rankacc_group2 = " + str(rankacc_esbased[-1]) + ", " + "spearman_group1 = " + str(spearman_esbased[-2]) + ", " + "spearman_group2 = " + str(spearman_esbased[-1]))  
        #xvecnorm = (np.dot(train_x,train_x.T).diagonal())**(1/2)
        #print(xvecnorm)
    else:
        break



