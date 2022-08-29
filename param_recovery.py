#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:38:42 2022

@author: s7340493
"""
#%%
import torch
import numpy as np
from scipy import io
import seaborn as sns
#from convenience_functions import read_pickle, write_pickle
from os import getcwd

from pybefit.agents import VISAT
from pybefit.inference import NormalGammaDiscreteDepth
from pybefit.tasks import SpaceAdventure

from simulate import Simulator
    

reppath = getcwd()

sns.set(context='talk', style='white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)

# define true parameter values for simulations as a list of parameter lists [alpha, beta, theta, depth] each configuring one agent.
# [alpha, beta, theta, depth]
simparams = [
    [0, 1, 0, 3],
    [0, 1.5, 0, 3],
    [0, 2, 0, 3]
]
#%% simulate behavior with different agents:

def simulate_SAT(params=[[0,1e10,0,3]], simruns=100):
    '''simulates space adventure task with agents according to params.
    
    Parameters:
        params (list): a list of parameter lists [alpha, beta, theta, depth] each configuring one agent.
        simruns (int): specifies how often each agent is simulated.
        
    Returns:
        simulations (list): list of Simulator objects
        performance (list): list of sum of point gain/loss for each individual agent, run, mini-block
    '''
    
    simulations = []
    performance = []
        
    # prepare environment
    exp = io.loadmat(reppath + '/experiment/experimental_variables_new.mat')
    starts = exp['startsExp'][:, 0] - 1
    planets = exp['planetsExp'] - 1
    vect = np.eye(5)[planets]
    
    # setup parameters for the task environment
    blocks = 100    # nr of mini-blocks
    runs = simruns  # nr of simulation runs
    ns = 6          # nr of possible states
    no = 5          # nr of planet types
    
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
    
    
    # iterate over agents configured by params and simulate
    
    for ps in params:
        
        # setup parameters for agent
        alpha, beta, theta, depth = ps
        depth = int(depth)
        trans_par = torch.tensor([beta, theta, alpha], dtype=torch.float32)
        trans_par = trans_par.repeat(runs,1)
        
        
        # define space adventure task with aquired configurations
        # set number of trials to the max number of actions
        space_advent = SpaceAdventure(conditions,
                                      outcome_likelihoods=confs,
                                      init_states=starts,
                                      runs=runs,
                                      mini_blocks=blocks,
                                      trials=3)
        
        # define the agent, each with a different maximal planning depth
        agent = VISAT(
            confs,
            runs=runs,
            mini_blocks=blocks,
            trials=3,
            planning_depth=depth
        )
        
        agent.set_parameters(trans_par, true_params=True)
        
        # simulate experiment
        sim = Simulator(space_advent, 
                        agent, 
                        runs=runs, 
                        mini_blocks=blocks,
                        trials=3)   # <- agent is internally always run for 3 trials!!!
        sim.simulate_experiment()
        
        simulations.append(sim)
            
        responses = sim.responses.clone()
        responses[torch.isnan(responses)] = 0
        responses = responses.long()
        
        outcomes = sim.outcomes
        
        points = costs[responses] + fuel[outcomes]
        points[outcomes<0] = 0
        performance.append(points.sum(-1))   # append sum of point gain/loss for each individual mini-block for given pl_depth
    
    # # dump simulations to disk
    # write_pickle(obj=simulations, relnm='sim_' + simname + '.pckl')
    
    return simulations, performance


simulations, performance = simulate_SAT(simparams)


#%% infer params from simulations

def load_and_format_behavioural_data(sim):
       
    runs = sim.agent.runs  # number of subjects
    
    responses = sim.responses
    states = sim.env.states
    conditions = sim.env.conditions
    confs = sim.env.ol
    ids = [i for i in range(runs)]
    mask = ~torch.isnan(responses)

    states[states < 0] = -1

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    return stimuli, mask, responses, conditions, ids

def variational_inference(stimuli, mask, responses):
    max_depth = 3
    mini_blocks, max_trials, runs = responses.shape
    
    confs = stimuli['configs']
    
    # define agent
    agent = VISAT(
        confs,
        runs=runs,
        mini_blocks=mini_blocks,
        trials=max_trials,
        planning_depth=max_depth
    )

    # load inference module and start model fitting
    infer = NormalGammaDiscreteDepth(agent, stimuli, responses, mask)
    infer.infer_posterior(iter_steps=2000, num_particles=10)
    
    return infer

def format_posterior_samples(infer):
    labels = [r'$\beta$', r'$\theta$', r'$\alpha$']
    _, pars_df, mg_df, sg_df = infer.sample_from_posterior(labels)

    return pars_df.melt(id_vars=['subject'], var_name='parameter')

def get_posterior_stats(post_marg, mini_blocks=100):
    n_samples, runs, max_trials = post_marg['d_0_0'].shape
    post_depth = {0: np.zeros((n_samples, mini_blocks, runs, max_trials)),
              1: np.zeros((n_samples, mini_blocks, runs, max_trials))}
    for pm in post_marg:
        b, t = np.array(pm.split('_')[1:]).astype(int)
        if t in post_depth:
            post_depth[t][:, b] = post_marg[pm]

    # get sample mean over planning depth for the first and second choice
    m_prob = [post_depth[c].mean(0) for c in range(2)]

    
    return post_depth, m_prob

def infer_from_simulation(sim):
    '''infers model params from Simulator object'''

    # load data
    stimuli, mask, responses, conditions, ids = load_and_format_behavioural_data(sim)

    # Variational inference
    infer = variational_inference(stimuli, mask, responses.permute(2, 1, 0))
    pars_df = format_posterior_samples(infer)

    max_trials = conditions[-1, :, 0]
    order = (max_trials == 3) + 1
    pars_df['order'] = order[pars_df.subject - 1]
    
    n_samples = 100
    post_marg = infer.sample_posterior_marginal(n_samples=n_samples)
    pars_df['IDs'] = np.array(ids)[pars_df.subject.values - 1]
    
    # store the value ELBO converged to
    loss = float(np.mean(infer.loss[-50:]))
    
    post_depth, m_prob = get_posterior_stats(post_marg)
    
    return pars_df, post_depth, m_prob, loss
    
# loop through simulations to recover params

info = "simulation, parameter, true_value\n" # headline of info-csv for param recovery

for i in range(len(simulations)):
    s = simulations[i]
    pars_df, post_depth, m_prob, loss = infer_from_simulation(s)

    # get sim params
    alpha = s.agent.alpha.detach().numpy().mean()
    beta = s.agent.beta.detach().numpy().mean()
    theta = s.agent.theta.detach().numpy().mean()
    depth = s.agent.depth

    # add parameter recovery info to info file
    info = info + "{0},alpha,{1}\n{0},beta,{2}\n{0},theta,{3}\n{0},depth,{4}\n".format(i,alpha,beta,theta,depth)

    # store samples of alpha,beta,theta
    pars_df.to_csv('paramrec{0}_pars_post_samples.csv'.format(i))

    # store planning depth samples of first action and meanPD per subject/run
    np.savez('paramrec{0}_PD_post_samples'.format(i),m_prob[0])
    ns = s.runs
    meanPD = np.matmul(m_prob[0], np.arange(1,4)).mean(0)
    np.savetxt('paramrec{0}_meanPD.csv'.format(i), np.stack([np.arange(1,ns+1),meanPD],axis=1), delimiter=",", header="subject, meanPD")

# store info file
with open("paramrec_info.csv","w") as csv_file:
    csv_file.write(info)




