# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:27:02 2019

@author: Howard
"""

import scipy.optimize as opt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from collections import Counter
import matplotlib.pyplot as plt
data = np.load("D:\\AISTATS\\Data\\car\\car_observation.npy")
#%% last
num_jou = 20
num_class = 3
num_worker = 60
#observation = []
#for i in range(len(data)):
#    if data[i,3]==1:
#        observation.append(data[i,1])
#    else:
#        observation.append(data[i,2])
#observation = np.reshape(np.array(observation),(len(observation),1))
#obs = np.column_stack((np.array(observation),data[:,0]))
#%% syn,con,jou

tri_upper_no_diag = np.triu(data, k=1)
obser_list = []
for k in range(num_worker):
    for i in range(num_jou):
        for j in range(num_jou):
            if j>i and tri_upper_no_diag[k,i,j]==1:
                obser_list.append([i,k])
            if j>i and tri_upper_no_diag[k,i,j]==0: 
                obser_list.append([j,k])
obs = np.array(obser_list)
#np.savetxt('car.txt', obs,fmt='%d',delimiter=',')
