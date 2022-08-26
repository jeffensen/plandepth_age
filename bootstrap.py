# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:23:54 2022

@author: s7340493
"""

from os import name, getcwd
import numpy as np
import pandas as pd
import random
from datetime import datetime
from scipy.stats import ttest_ind, bootstrap
import matplotlib.pyplot as plt
from math import sqrt

random.seed(datetime.now())

# CONSTANTS
n_boot = 10000
vboot = "Model_alpha"
vgr  = "Group"
ci_per = .95 # percentage in CI-interval (between 0-1)


# Load data
currentdir = getcwd()
if name == 'nt':
    #windows paths
    datadir = currentdir + '\\data\\'
if name == 'posix':
    #linux paths
    datadir = currentdir + '/data/'
    
datafile = datadir + "SAT_subjectLevel_and_cognTasks.xlsx"
df = pd.read_excel(datafile)
dfVgr0 = df[df[vgr]==0][vboot].to_numpy()
dfVgr1 = df[df[vgr]==1][vboot].to_numpy()
dfVall = df[vboot].to_numpy()
ngr0 = len(dfVgr0)
ngr1 = len(dfVgr1)


# get actual mean diff
gr0mean = dfVgr0.mean()
gr1mean = dfVgr1.mean()
effect = gr0mean - gr1mean
se = sqrt(dfVgr0.std()**2 /ngr0 + dfVgr1.std()**2 /ngr1)
t = effect / se

# shift samples such that H0 is true
overallmean = dfVall.mean()
dfVgr0Shifted = dfVgr0 - gr0mean + overallmean
dfVgr1Shifted = dfVgr1 - gr1mean + overallmean
dfVallShifted = np.concatenate((dfVgr0Shifted, dfVgr1Shifted))


# draw bootstrap samples with scipy (does not use the shifted values, i.e. directly computes CI around true sample effect)
#   define function
def meandiff(sample1, sample2):
    meandiff = sample1.mean() - sample2.mean()
    return meandiff
bs_sci = bootstrap((dfVgr0,dfVgr1), meandiff, confidence_level=0.95, n_resamples=n_boot, vectorized=False, method="percentile")
ci_sci = list(bs_sci.confidence_interval)


# draw boot samples
boots = []
boots_t = []
for i in range(n_boot):
    boot0 = np.random.choice(dfVallShifted, size=ngr0+ngr1, replace = True)
    boot1 = boot0[ngr0:]
    boot0 = boot0[0:ngr0]
    boots.append(boot0.mean() - boot1.mean())
    boots_t.append((boot0.mean() - boot1.mean()) / (sqrt(boot0.std()**2 /ngr0 + boot1.std()**2 /ngr1))) 

# sort to get ci and p-value
boots.sort()
l = int(n_boot*(1-ci_per)/2)
r = n_boot-l
ciNull = [boots[l],boots[r]]
se_boot = np.std(boots)
ci_boot = [effect-1.96*se_boot, effect+1.96*se_boot]

boots_abs = list(map(abs, boots))
n_asextreme = len(list(filter(lambda diff: diff >=  abs(effect), boots_abs)))
p = n_asextreme / n_boot

boots_t_abs = list(map(abs, boots_t))
n_t_asextreme = len(list(filter(lambda diff: diff >=  abs(t), boots_t_abs)))
p_t = n_t_asextreme / n_boot

