# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:36:20 2020

@author: Johannes
"""
#%%

import torch
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import seaborn as sns

from convenience_functions import read_pickle, write_pickle

from pybefit.tasks import SpaceAdventure
from pybefit.agents import VISAT
from simulate import Simulator

from local_agents import MFRL
from simulate import SimulatorMFRL


from os import name, getcwd
reppath = getcwd()

sns.set(context='talk', style='white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)
phasecolors = ['mediumaquamarine',
               'g',
               'skyblue',
               'b']
noisecolors = ['skyblue','b']


#%% simulate behavior with different agents:

agents = []
simulations = []
performance = []
simnames = []
resps = []

# prepare environment
exp = io.loadmat(reppath + '/experiment/experimental_variables_new.mat')
starts = exp['startsExp'][:, 0] - 1
planets = exp['planetsExp'] - 1
vect = np.eye(5)[planets]

# setup parameters for the task environment
blocks = 100 # nr of mini-blocks
runs = 1000  # nr of simulation runs
ns = 6       # nr of possible states
no = 5       # nr of planet types
na = 2       # nr of actions

ol1 = torch.from_numpy(vect)                              # planet configurations for order=1
ol2 = torch.from_numpy(np.vstack([vect[50:], vect[:50]])) # planet configurations for order=2

starts1 = torch.from_numpy(starts)                                # start planet for order=1
starts2 = torch.from_numpy(np.hstack([starts[50:], starts[:50]])) # start planet for order=2
    
noise = np.tile(np.array([0, 1, 0, 1]), (25,1)).T.flatten()   # mini-blocks with noise condition?
trials1 = np.tile(np.array([2, 2, 3, 3]), (25,1)).T.flatten() # nr of trials for order=1
trials2 = np.tile(np.array([3, 3, 2, 2]), (25,1)).T.flatten() # nr of trials for order=2

costs = torch.FloatTensor([-2, -5])  # action costs
fuel = torch.arange(-20., 30., 10.)  # fuel reward of each planet type

# tensor of configurations for all runs
# with first half of runs order=1 and second half order=2
confs = torch.stack([ol1, ol2])
confs = confs.view(2, 1, blocks, ns, no).repeat(1, runs//2, 1, 1, 1)\
        .reshape(-1, blocks, ns, no).float()

starts = torch.stack([starts1, starts2])
starts = starts.view(2, 1, blocks).repeat(1, runs//2, 1)\
        .reshape(-1, blocks)

# tensor of conditions for all runs
# conditions[0] for noise condition
# conditions[1] for nr of trials condition        
conditions = torch.zeros(2, runs, blocks, dtype=torch.long)
conditions[0] = torch.tensor(noise, dtype=torch.long)[None,:]
conditions[1, :runs//2] = torch.tensor(trials1, dtype=torch.long)
conditions[1, runs//2:] = torch.tensor(trials2, dtype=torch.long)

#%% Optimal agents with varying planning depth
simname = 'optimal'

# setup parameters for agent
beta  = 1e10 
alpha = 0.0
theta = 0.0
trans_par = torch.tensor([beta, theta, alpha], dtype=torch.float)
trans_par = trans_par.repeat(runs,1)


# iterate over different planning depths
for depth in range(2, -1, -1):
    
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions
    space_advent = SpaceAdventure(conditions,
                                  outcome_likelihoods=confs,
                                  init_states=starts,
                                  runs=runs,
                                  mini_blocks=blocks,
                                  trials=3)
    
    # define the optimal agent, each with a different maximal planning depth
    agent = VISAT(
        confs,
        runs=runs,
        mini_blocks=blocks,
        trials=3,
        planning_depth=depth+1
    )
    
    agent.set_parameters(trans_par)
    
    # simulate experiment
    sim = Simulator(space_advent, 
                    agent, 
                    runs=runs, 
                    mini_blocks=blocks,
                    trials=3)   # <- agent is internally always run for 3 trials!!!
    sim.simulate_experiment()
    
    simulations.append(sim)
    agents.append(agent)
        
    responses = sim.responses.clone()
    responses[torch.isnan(responses)] = 0
    responses = responses.long()
    resps.append(responses)
    
    outcomes = sim.outcomes
    
    points = costs[responses] + fuel[outcomes]
    points[outcomes<0] = 0
    performance.append(points.sum(-1))   # append sum of point gain/loss for each individual mini-block for given pl_depth

    simnames.append('depth-'+str(depth+1))
# # dump simulations to disk
# write_pickle(obj=simulations, relnm='sim_' + simname + '.pckl')


#%% simulate behavior with Model-free RL Agent
simname = 'MF-RL'

# # iterate over parameter values to find maximum performance
# alphas = np.arange(0,1,0.1).tolist()
# epsilons = np.arange(0,0.5,0.05).tolist()

# for alpha in alphas:
#     for epsilon in epsilons:
            
# setup parameters for agent
alpha = 0.5
epsilon = 0.3
pars = torch.tensor([alpha, epsilon], dtype=torch.float)
pars = pars.repeat(runs,1)

agentname = simname + "_a-" + str(alpha) + "_e-" + str(epsilon)

# define space adventure task with aquired configurations
# set number of trials to the max number of actions
space_advent = SpaceAdventure(conditions,
                              outcome_likelihoods=confs,
                              init_states=starts,
                              runs=runs,
                              mini_blocks=blocks,
                              trials=3)

# define the optimal agent, each with a different maximal planning depth
agent = MFRL(confs,
            runs=runs,
            mini_blocks=blocks,
            trials=3)

agent.set_parameters(pars)

# simulate experiment
sim = SimulatorMFRL(space_advent, 
                    agent, 
                    runs=runs, 
                    mini_blocks=blocks,
                    trials=3)   # <- agent is internally always run for 3 trials!!!
sim.simulate_experiment()

simulations.append(sim)
agents.append(agent)


responses = sim.responses.clone()
responses[torch.isnan(responses)] = 0
responses = responses.long()
resps.append(responses)

outcomes = sim.outcomes

points = costs[responses] + fuel[outcomes]
points[outcomes<0] = 0
performance.append(points.sum(-1))   # append sum of point gain/loss for each individual mini-block for given pl_depth

simnames.append(simname)

# # dump simulations to disk
# write_pickle(obj=simulations, relnm='sim_' + simname + '.pckl')

  

#%% agents with random action selection
simname = 'random'

# setup parameters for agent
beta  = 1e-10 # <- approx. random action selection
alpha = 0.0
theta = 0.0
trans_par = torch.tensor([beta, theta, alpha], dtype=torch.float)
trans_par = trans_par.repeat(runs,1)
    
# define space adventure task with aquired configurations
# set number of trials to the max number of actions
space_advent = SpaceAdventure(conditions,
                              outcome_likelihoods=confs,
                              init_states=starts,
                              runs=runs,
                              mini_blocks=blocks,
                              trials=3)

# define the optimal agent, each with a different maximal planning depth
agent = VISAT(
    confs,
    runs=runs,
    mini_blocks=blocks,
    trials=3,
    planning_depth=depth+1
)

agent.set_parameters(trans_par)

# simulate experiment
sim = Simulator(space_advent, 
                agent, 
                runs=runs, 
                mini_blocks=blocks,
                trials=3)   # <- agent is internally always run for 3 trials!!!
sim.simulate_experiment()

simulations.append(sim)
agent.depth=np.nan
agents.append(agent)
    
responses = sim.responses.clone()
responses[torch.isnan(responses)] = 0
responses = responses.long()
resps.append(responses)

outcomes = sim.outcomes

points = costs[responses] + fuel[outcomes]
points[outcomes<0] = 0
performance.append(points.sum(-1))   # append sum of point gain/loss for each individual mini-block for given pl_depth

simnames.append(simname)

# # dump simulations to disk
# write_pickle(obj=simulations, relnm='sim_' + simname + '.pckl')    

#%% store sim data
sim_data = pd.DataFrame()

nAg = len(agents)

scores = []
rel_score = np.zeros((nAg, 4))

start_points = 1000

for ag in range(nAg):
    end_points = start_points + performance[ag].numpy().cumsum(-1)
    points = np.hstack([start_points*np.ones((runs, 1)), end_points])
    df = pd.DataFrame(points.T, columns=range(runs))
    df['mini-block'] = np.arange(blocks + 1)
    df = pd.melt(df, id_vars='mini-block', value_vars=range(runs), value_name='points')
    df['order'] = 1
    df.loc[df.variable >= runs//2, 'order'] = 2
    df['agent'] = simnames[ag]
    df['gain'] = df.points.diff()
    df.loc[df['mini-block']==0,'gain'] = np.nan
    df['phase'] = ((df['mini-block']-1)//25 + 1 )
    df.loc[df['phase']==0, 'phase'] = 1
    
    # set mini-block and phase conditioned on order
    df['phase_id'] = df.phase
    df.loc[df['order']==2, 'phase_id'] = df.phase_id-2
    df.loc[df['phase_id']<=0, 'phase_id'] = df.phase_id+4
    df['mini-block_id'] = df['mini-block']
    df.loc[df['order']==2, 'mini-block_id'] = df['mini-block_id']-50
    df.loc[(df['mini-block_id']<=0) & (df['order']==2), 'mini-block_id'] = df['mini-block_id']+100
    df.loc[df['mini-block_id']>blocks, 'mini-block_id'] = blocks
    df.loc[df['mini-block']==0, 'mini-block_id'] = 0
    
    
    if hasattr(agents[ag],'depth'):
        df['depth'] = agents[ag].depth
    if hasattr(agents[ag],'epsilon') & hasattr(agents[ag],'alpha'):
        df['alpha'] = agents[ag].alpha[0].item()
        df['epsilon'] = agents[ag].epsilon[0].item()
    
    sim_data = sim_data.append(df, ignore_index=True)
    
    phase_values = np.concatenate([start_points*np.ones((runs, 1)), end_points.reshape(-1, 4, 25)[..., -1]], -1)
    diffs = np.diff(phase_values, axis=-1)
                    
    diffs[runs//2:, :] = np.concatenate([diffs[runs//2:, 2:], diffs[runs//2:, :2]], -1)
    scores.append(diffs/25)
    
    rel_score[ag] = diffs.mean(0)/25


#%% plot agents points over mb
plt.figure
g = sns.FacetGrid(sim_data, col="order", hue="agent", height=5)
g.map(sns.lineplot, "mini-block", "points").add_legend();
g.axes.flat[0].set_title("normal order")
g.axes.flat[1].set_title("reversed order")
plt.savefig('sim_' + simname + '_pointsplot', dpi=300)
plt.savefig('sim_' + simname + '_pointsplot.svg', format="svg")


#%% MF-RL parameter optimization: heatmap of final scores for each parameter combination
sim_data2 = sim_data[sim_data['mini-block']==99].groupby('agent').mean()

sim_data2.alpha = sim_data2.alpha.astype("category")
sim_data2.epsilon = sim_data2.epsilon.astype("category")

heatmap1_data = pd.pivot_table(sim_data2, values='points', 
                     index=['alpha'], 
                     columns='epsilon')
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(heatmap1_data, ax = ax, cbar_kws={'label': 'final point score'})
# ax.set_xticklabels([str("%.2f" % round(i,2)) for i in epsilons])
# ax.set_yticklabels([str("%.1f" % round(i,2)) for i in alphas])

fig.savefig('sim_MF-RL-score-heatmap.png', dpi=300)
fig.savefig('sim_MF-RL-score-heatmap.svg', type="svg")


#%% Plot GainMEAN per Phase and Depth
phasecol = 'phase_id'
blockcol = 'mini-block_id'
sim_data2 = sim_data[(sim_data['depth'].notna()) & (sim_data[blockcol]>0)].groupby(['agent',blockcol]).mean().reset_index()
sim_data2.depth = sim_data2.depth.astype('int')
sim_data2[phasecol] = sim_data2[phasecol].astype('int')

depths = sim_data2.depth.unique()
n = len(depths)
fig, ax = plt.subplots(n, 1, sharex=True, sharey= True, figsize=(15,15))
fig.suptitle('Mean Agent\'s points gain per Phase and pd', fontsize=30)
for a in range(n):
    sns.boxplot(data = sim_data2[sim_data2['depth']==depths[a]],
                x = 'gain',
                y = phasecol,
                ax = ax[a],
                palette = phasecolors,
                orient='h')
    ax[a].set_xlabel('points gain | pd = '+str(depths[a]))
    #ax[a].axhline(color = 'black')
plt.show()
fig.savefig('GainMEANbox.png', dpi=300)
fig.savefig('GainMEANbox.svg', type='svg')
plt.close()

