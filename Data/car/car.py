# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:50:36 2020

@author: Howard
"""
import pandas as pd
import numpy as np
from collections import Counter
num_worker = 60
num_item = 20
worker_list = np.arange(num_worker)
preference = pd.read_csv('D:\\AISTATS\\Data\\car\\prefs2.csv').values
#%%
true_c = pd.read_csv('D:\\AISTATS\\Data\\car\\items2.csv').values[:,1]
oberservation = np.zeros((num_worker,num_item,num_item))
for i in range(len(preference)):
    for j in range(num_worker):
        if preference[i,0]-1 == worker_list[j]:
            if preference[i,3]==0:
                oberservation[j,preference[i,2]-1,preference[i,1]-1] = 1
            else:
                oberservation[j,preference[i,1]-1,preference[i,2]-1] = 1
obs = np.ones((num_worker,num_item,num_item))*-1
for i in range(num_worker):
    for j in range(num_item):
        for k in range(num_item):
            if oberservation[i,j,k]==1 and k>j:
                obs[i,j,k] = 0
            if oberservation[i,j,k]==1 and j>k:
                obs[i,k,j] = oberservation[i,j,k]
                
np.save("car_observation.npy",obs)                

#%%
win_list = []
lose_list = []
for i in range(len(preference)):
    if preference[i,3]==0:
        win_list.append(preference[i,1])
        lose_list.append(preference[i,2])
    else:
        win_list.append(preference[i,2])
        lose_list.append(preference[i,1])
win_count = Counter(win_list).most_common()
lose_count = Counter(lose_list).most_common()
score_list = []
for i in range(num_item):
    for j in range(num_item):
        if win_count[i][0]==lose_count[j][0] :
            score_list.append([win_count[i][0],win_count[i][1]/(lose_count[i][1]+win_count[i][1])])
score = np.array(score_list)   
df = pd.DataFrame(data=score)        
true_s = df.sort_values(by=[0]).values[:,1]        
#np.save("car_s",true_s)
#np.save("car_c",true_c)
