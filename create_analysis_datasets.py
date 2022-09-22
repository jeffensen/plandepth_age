# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:49:48 2022

@author: jstef
"""

from scipy import io
import pandas as pd
from numpy import zeros
import numpy as np
from os import getcwd, path, listdir
from utils import map_noise_to_values

# set data directory as relative path from this files directory
reppath = getcwd()
datadir = path.join(reppath,"data")
    

def load_and_format_SAT_data(datadir):
    '''
    Parameters
    ----------
    datadir : string
        Full os-compatible path to directory containing SAT result files.
        Filename format: *_ID*.mat, i.e. subject id must follow first underscore.

    Returns
    -------
    stimuli:    dict(3) of numpy.ndarrays for task conditions, -configs, -states
    responses:  numpy.ndarray(sbj,mb,trial) of responses 
    mask:       numpy.ndarray(sbj,mb,trial) of non-NaN value indicators for responses
    rts:        numpy.ndarray(sbj,mb,trial) of reaction times
    scores:     numpy.ndarray(sbj,mb,trial) of point scores
    conditions: numpy.ndarray(noise/steps,sbj,mb) of task conditions
    ids:        list of ids
    sbj_df:     pandas.DataFrame rows:sbj, cols:ID,age,gender,group
    '''
    
    fnames = listdir(datadir)
    
    runs = len(fnames)  # number of subjects
    
    mini_blocks = 100  # number of mini blocks in each run
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth
    no = 5 # number of outcomes/rewards
    max_points = 2009 # maximum possible total points (out of 100.000 depth3-agent simulations)

    responses = zeros((runs, mini_blocks, max_trials), dtype=int)
    states = zeros((runs, mini_blocks, max_trials+1), dtype=int)
    scores = zeros((runs, mini_blocks, max_depth), dtype=int)
    conditions = zeros((2, runs, mini_blocks), dtype=int)
    confs = zeros((runs, mini_blocks, 6), dtype=int)
    rts = zeros((runs, mini_blocks, 3), dtype=np.float64)
    ids = []
    age = []
    gender = []
    total_points = []
    performance = []
    for i,f in enumerate(fnames):
        parts = f.split('_')
        tmp = io.loadmat(path.join(datadir,f))
        responses[i] = tmp['data']['Responses'][0,0]['Keys'][0,0]-1
        states[i] = (tmp['data']['States'][0,0] - 1)
        confs[i] = (tmp['data']['PlanetConf'][0,0] - 1)
        scores[i] = tmp['data']['Points'][0,0]
        strings = tmp['data']['Conditions'][0,0]['noise'][0,0][0]
        conditions[0, i] = list(map_noise_to_values(strings))
        conditions[1, i] = (tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0])
        rts[i] = tmp['data']['Responses'][0,0]['RT'][0,0]
        ids.append(parts[1])
        age.append(tmp['data']['Age'][0,0][0,0])
        gender.append(tmp['data']['Gender'][0,0][0,0])
        if conditions[1,i,-1]==3:
            total_points.append(scores[i,-1,-1])
        else:
            total_points.append(scores[i,-1,-2])
        performance.append(total_points[-1] / max_points * 100)
        
    sbj_df = pd.DataFrame(list(zip(ids,age,gender)),columns=['ID','age','gender'])

    states[states < 0] = -1
    confs = np.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~np.isnan(responses)
    
    return stimuli, conditions, responses, mask, rts, scores, total_points, performance, ids, sbj_df


def load_and_format_IDP_or_SAW_data(datadir,taskname):
    
    t = taskname
    fnames = listdir(datadir)
    ids = []
    
    df = pd.DataFrame(columns=['ID',t+'_CORR',t+'_MaxCORR',t+'_PER',t+'_ERR',t+'_ACC',t+'_RT',t+'_RT_SD'])

    for i,f in enumerate(fnames):
        parts = f.split('_')
        ids.append(parts[0])
        
        tmp = pd.read_csv(path.join(datadir,f), sep='\t', dtype={'VPID':object})
        maxtrials = 46 if t=='IDP' else 35 if t=='SAW' else print("WARNING: unknown value for taskname!")
        if len(tmp) < maxtrials: print("WARNING: Trials missing in file of ID "+str(ids[-1]))
        
        # delete rows after deadline was exceeded ('RESP' = 0)
        tmp = tmp.loc[tmp['RESP']>0]
        # exclude RTs < 150ms
        tmp = tmp.loc[tmp['RESPT']>=150]
        
        ntrials = len(tmp) # n of finished trials
        ncorr = len(tmp.loc[tmp['ACC']>0]) # n of correct trials  
        
        # create row for subject-wise dataframe
        entry = pd.DataFrame(columns=['ID',t+'_CORR',t+'_MaxCORR',t+'_PER',t+'_ERR',t+'_ACC',t+'_RT',t+'_RT_SD'])
        entry['ID']          = [tmp['VPID'].iat[0]]
        entry[t+'_CORR']    = [ncorr]
        entry[t+'_MaxCORR'] = [maxtrials]
        entry[t+'_PER']     = [ncorr / maxtrials * 100]
        entry[t+'_ERR']     = [ntrials - ncorr]
        entry[t+'_ACC']     = [ncorr / ntrials * 100]
        entry[t+'_RT']      = [tmp['RESPT'].mean() / 1000 ]
        entry[t+'_RT_SD']   = [tmp['RESPT'].std()  / 1000 ]
        
        df = pd.concat([df, entry], ignore_index=True)
    
    return df


def load_and_format_SWM_data(datadir):
    
    # function only focuses on the first task condition (location memory condition) across both load levels (4 and 7 items)
    
    fnames = listdir(datadir)
    ids = []
    
    df = pd.DataFrame(columns=['ID','SWM_CORR','SWM_MaxCORR','SWM_PER','SWM_ERR','SWM_ACC','SWM_RT','SWM_RT_SD'])

    for i,f in enumerate(fnames):
        parts = f.split('_')
        ids.append(parts[0])
        
        tmp = pd.read_csv(path.join(datadir,f), sep='\t', dtype={'VPID':object}, index_col=False) # index_col=False to prevent confusion because files contain delimiters at end of lines
        maxtrials = 96
        if len(tmp) < maxtrials: print("WARNING: Trials missing in file of ID "+str(ids[-1]))
        
        # drop first 4 rows which contain the data of the 4 training trials during instruction
        tmp.drop(tmp.index[:4], inplace=True)
        # delete rows after deadline was exceeded ('Resp1' = -999)
        tmp = tmp.loc[tmp['Resp1']>=0]
        # exclude RTs < 150ms
        tmp = tmp.loc[tmp['RT1']>=150]
        
        ncorr = len(tmp.loc[tmp['Corr1']>0]) # n of correct trials
        ntrials = len(tmp) # n of finished trials
        
        # create row for subject-wise dataframe
        entry = pd.DataFrame(columns=['ID','SWM_CORR','SWM_MaxCORR','SWM_PER','SWM_ERR','SWM_ACC','SWM_RT','SWM_RT_SD'])
        entry['ID']          = [tmp['VPID'].iat[0]]
        entry['SWM_CORR']    = [ncorr]
        entry['SWM_MaxCORR'] = [maxtrials]
        entry['SWM_PER']     = [ncorr / maxtrials * 100]
        entry['SWM_ERR']     = [ntrials - ncorr]
        entry['SWM_ACC']     = [ncorr / ntrials * 100]
        entry['SWM_RT']      = [tmp['RT1'].mean() / 1000 ]
        entry['SWM_RT_SD']   = [tmp['RT1'].std()  / 1000 ]
        
        df = pd.concat([df, entry], ignore_index=True)
    
    return df

#%% load and format behavioural data
# SAT raw data
path_ya =  path.join(datadir,"YoungAdults","space_adventure","Experiment")
path_oa =  path.join(datadir,"OldAdults","space_adventure","Experiment")

stimuli_ya, conditions_ya, responses_ya, mask_ya, rts_ya, scores_ya, total_points_ya, per_ya, ids_ya, sbj_df_ya = load_and_format_SAT_data(path_ya)
stimuli_oa, conditions_oa, responses_oa, mask_oa, rts_oa, scores_oa, total_points_oa, per_oa, ids_oa, sbj_df_oa = load_and_format_SAT_data(path_oa)

sbj_df_ya['group'] = 0
sbj_df_oa['group'] = 1

# IDP and SAW data
idp_df_oa = load_and_format_IDP_or_SAW_data(path.join(datadir,'OldAdults','Identical Pictures'),'IDP')
idp_df_ya = load_and_format_IDP_or_SAW_data(path.join(datadir,'YoungAdults','Identical Pictures'),'IDP')

saw_df_oa = load_and_format_IDP_or_SAW_data(path.join(datadir,'OldAdults','Spot a Word'),'SAW')
saw_df_ya = load_and_format_IDP_or_SAW_data(path.join(datadir,'YoungAdults','Spot a Word'),'SAW')

# SWM data
swm_df_oa = load_and_format_SWM_data(path.join(datadir,'OldAdults','Spatial Working Memory'))
swm_df_ya = load_and_format_SWM_data(path.join(datadir,'YoungAdults','Spatial Working Memory'))

# merge all dfs
sbj_df_ya = sbj_df_ya.merge(idp_df_ya, on='ID', validate="one_to_one")
sbj_df_ya = sbj_df_ya.merge(saw_df_ya, on='ID', validate="one_to_one")
sbj_df_ya = sbj_df_ya.merge(swm_df_ya, on='ID', validate="one_to_one")
sbj_df_oa = sbj_df_oa.merge(idp_df_oa, on='ID', validate="one_to_one")
sbj_df_oa = sbj_df_oa.merge(saw_df_oa, on='ID', validate="one_to_one")
sbj_df_oa = sbj_df_oa.merge(swm_df_oa, on='ID', validate="one_to_one")


#%% load SAT inference results

# load param posterior samples
pars_df = pd.read_csv('pars_post_samples.csv', index_col=0, dtype={'IDs':object})
pars_df.rename(columns={'IDs':'ID'}, inplace=True)
pars_df.rename(columns={'group':'group_label'}, inplace=True)
# get order if IDs of inference results
pars_IDorder_oa = pars_df[pars_df.group_label=='OA'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()
pars_IDorder_ya = pars_df[pars_df.group_label=='YA'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()

# load depth posterior stats
tmp = np.load("oa_plandepth_stats.npz", allow_pickle=True)
m_prob_oa = tmp['arr_1']
tmp = np.load("ya_plandepth_stats.npz", allow_pickle=True)
m_prob_ya = tmp['arr_1']


#%% create and store dataframes on miniblock-, condition- and subject-level as csv for further analyses

# create miniblock-wise dfs
mini_blocks = 100
nsub_oa = sbj_df_oa.shape[0]
m_prob_oa_IDindex = (pars_IDorder_oa.repeat(mini_blocks)-1)*mini_blocks + np.tile(np.arange(0,mini_blocks),nsub_oa)

mb_df_oa = pd.concat([sbj_df_oa] * mini_blocks, ignore_index=True).sort_values(by="ID", ignore_index=True)
mb_df_oa["block_num"] = np.tile(np.arange(1,mini_blocks+1),nsub_oa)
mb_df_oa["noise"] = conditions_oa[0,:,:].reshape(mini_blocks * nsub_oa)
mb_df_oa["steps"] = conditions_oa[1,:,:].reshape(mini_blocks * nsub_oa)
mb_df_oa["SAT_RT"] = rts_oa[:,:,0].reshape(mini_blocks * nsub_oa)
mb_df_oa['MeanPD'] = np.matmul(m_prob_oa[0], np.arange(1,4)).reshape(mini_blocks * nsub_oa, order='F')[m_prob_oa_IDindex]
mb_df_oa['SAT_Total_points'] = np.repeat(np.array(total_points_oa),100)
mb_df_oa['SAT_PER'] = np.repeat(np.array(per_oa),100)

nsub_ya = sbj_df_ya.shape[0]
m_prob_ya_IDindex = (pars_IDorder_ya.repeat(mini_blocks)-1)*mini_blocks + np.tile(np.arange(0,mini_blocks),nsub_ya)

mb_df_ya = pd.concat([sbj_df_ya] * mini_blocks, ignore_index=True).sort_values(by="ID", ignore_index=True)
mb_df_ya["block_num"] = np.tile(np.arange(1,mini_blocks+1),nsub_ya)
mb_df_ya["noise"] = conditions_ya[0,:,:].reshape(mini_blocks * nsub_ya)
mb_df_ya["steps"] = conditions_ya[1,:,:].reshape(mini_blocks * nsub_ya)
mb_df_ya["SAT_RT"] = rts_ya[:,:,0].reshape(mini_blocks * nsub_ya)
mb_df_ya['MeanPD'] = np.matmul(m_prob_ya[0], np.arange(1,4)).reshape(mini_blocks * nsub_ya, order='F')[m_prob_ya_IDindex]
mb_df_ya['SAT_Total_points'] = np.repeat(np.array(total_points_ya),100)
mb_df_ya['SAT_PER'] = np.repeat(np.array(per_ya),100)

SAT_singleMiniblocks_df = mb_df_oa.append(mb_df_ya, ignore_index=True)

# create subject-wise df of model params
pars_df_subj = pd.pivot(pars_df.groupby(["ID","group_label","parameter"], as_index=False).mean(),index=["ID","subject","group_label","order"],columns="parameter", values="value").reset_index().rename_axis(None, axis=1)
pars_df_subj.rename(columns={"$\\alpha$":"model_alpha","$\\beta$":"model_beta","$\\theta$":"model_theta"},inplace=True)

# add params and order to miniblock-wise dfs
SAT_singleMiniblocks_df = pd.merge(left=SAT_singleMiniblocks_df, right=pars_df_subj, on='ID')
SAT_singleMiniblocks_df['block_id'] = SAT_singleMiniblocks_df['block_num']
SAT_singleMiniblocks_df.loc[SAT_singleMiniblocks_df.order == 2, 'block_id'] = (SAT_singleMiniblocks_df.block_id + 49) % 100 + 1

# create condition-wise df
SAT_conditionLevel_df = SAT_singleMiniblocks_df.groupby(by=['ID','noise','steps'], as_index=False).mean()
SAT_conditionLevel_df.drop(columns=['block_num','block_id'], inplace=True)

# create subject-wise df
SAT_subjectLevel_df = SAT_singleMiniblocks_df.groupby(by=['ID'], as_index=False).mean()
SAT_subjectLevel_df.drop(columns=['noise','steps','block_num','block_id'], inplace=True)

# store all dfs
SAT_singleMiniblocks_df.to_csv('SAT_singleMiniblocks.csv')
SAT_conditionLevel_df.to_csv('SAT_conditionLevel.csv')
SAT_subjectLevel_df.to_csv('SAT_subjectLevel.csv')