import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from torch import zeros, ones
from os import listdir, path
from scipy import io

def errorplot(*args, **kwargs):
    """plot asymetric errorbars"""
    subjects = args[0]
    values = args[1].values
    
    unique_subjects = np.unique(subjects)
    nsub = len(unique_subjects)
    
    values = values.reshape(-1, nsub)
    
    quantiles = np.percentile(values, [5, 50, 95], axis=0)
    
    low_perc = quantiles[0]
    up_perc = quantiles[-1]
    
    x = unique_subjects
    y = quantiles[1]

    assert np.all(low_perc <= y)
    assert np.all(y <= up_perc)
    
    kwargs['yerr'] = [y-low_perc, up_perc-y]
    kwargs['linestyle'] = ''
    kwargs['marker'] = 'o'
    
    plt.errorbar(x, y, **kwargs)
    
def map_noise_to_values(strings):
    """mapping strings ('high', 'low') to numbers 0, 1"""
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nan

def load_and_format_behavioural_data(datadir):

    fnames = listdir(datadir)
    
    runs = len(fnames)  # number of subjects
    
    mini_blocks = 100  # number of mini blocks in each run
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth

    na = 2  # number of actions
    ns = 6 # number of states/locations
    no = 5 # number of outcomes/rewards

    responses = zeros(runs, mini_blocks, max_trials)
    states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
    scores = zeros(runs, mini_blocks, max_depth)
    conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
    confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
    rts = zeros(runs, mini_blocks, 3, dtype=torch.float64)
    ids = []
    age = []
    gender = []
    for i,f in enumerate(fnames):
        parts = f.split('_')
        tmp = io.loadmat(path.join(datadir,f))
        responses[i] = torch.from_numpy(tmp['data']['Responses'][0,0]['Keys'][0,0]-1)
        states[i] = torch.from_numpy(tmp['data']['States'][0,0] - 1).long()
        confs[i] = torch.from_numpy(tmp['data']['PlanetConf'][0,0] - 1).long()
        scores[i] = torch.from_numpy(tmp['data']['Points'][0,0])
        strings = tmp['data']['Conditions'][0,0]['noise'][0,0][0]
        conditions[0, i] = torch.tensor(list(map_noise_to_values(strings)), dtype=torch.long)
        conditions[1, i] = torch.from_numpy(tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]).long()
        rts[i] = torch.from_numpy(tmp['data']['Responses'][0,0]['RT'][0,0])
        ids.append(parts[1])
        age.append(tmp['data']['Age'][0,0][0,0])
        gender.append(tmp['data']['Gender'][0,0][0,0])
        
    sbj_df = pd.DataFrame(list(zip(ids,age,gender)),columns=['IDs','age','gender'])

    states[states < 0] = -1
    confs = torch.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~torch.isnan(responses)
    
    return stimuli, mask, responses, conditions, rts, ids, sbj_df
