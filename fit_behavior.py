#%% import pakages, methods and set environment
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import getcwd, path

# local classes and methods
from pybefit.agents.torch import VISAT  # value iteration agent for SAT
from pybefit.inference.pyro import NormalGammaDiscreteDepth  # A hierarchical parametric model with discrete variables
from utils import errorplot, load_and_format_behavioural_data

# set data directory as relative path from this files directory
reppath = getcwd()
datadir = path.join(reppath,"data")

sns.set(context='talk', style='white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)

#%% load and format behavioural data
path1 =  path.join(datadir,"YoungAdults","space_adventure","Experiment")
path2 =  path.join(datadir,"OldAdults","space_adventure","Experiment")

stimuli_ya, mask_ya, responses_ya, conditions_ya, rts_ya, ids_ya, sbj_df_ya = load_and_format_behavioural_data(path1)
stimuli_oa, mask_oa, responses_oa, conditions_oa, rts_oa, ids_oa, sbj_df_oa = load_and_format_behavioural_data(path2)

#%% Define probabilistic inference methods
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
    infer.infer_posterior(iter_steps=2000, num_particles=10, optim_kwargs={'lr': 0.01})
    
    return infer


# sample from posterior
def format_posterior_samples(infer):
    labels = [r'$\beta$', r'$\theta$', r'$\alpha$']
    _, pars_df, mg_df, sg_df = infer.sample_posterior(labels)

    return pars_df.melt(id_vars=['subject'], var_name='parameter')


#%% Variational inference OA
# passed responses should have the shape n_blocks, n_trials, n_subjects
infer_oa = variational_inference(stimuli_oa, mask_oa, responses_oa.permute(1, 2, 0))
pars_df_oa = format_posterior_samples(infer_oa)

max_trials = conditions_oa[-1, :, 0]
order_oa = (max_trials == 3) + 1
pars_df_oa['order'] = order_oa[pars_df_oa.subject - 1]

n_samples = 100
post_marg_oa = infer_oa.sample_posterior_marginal(n_samples=n_samples)
pars_df_oa['IDs'] = np.array(ids_oa)[pars_df_oa.subject.values - 1]


#%% Variational inference YA
# passed responses should have the shape n_blocks, n_trials, n_subjects
infer_ya = variational_inference(stimuli_ya, mask_ya, responses_ya.permute(1, 2, 0))
pars_df_ya = format_posterior_samples(infer_ya)

max_trials = conditions_ya[-1, :, 0]
order_ya = (max_trials == 3) + 1
pars_df_ya['order'] = order_ya[pars_df_ya.subject - 1]

n_samples = 100
post_marg_ya = infer_ya.sample_posterior_marginal(n_samples=n_samples)
pars_df_ya['IDs'] = np.array(ids_ya)[pars_df_ya.subject.values - 1]

#%% plot convergence of ELBO bound (approximate value of the negative marginal log likelihood)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
loss = [infer_oa.loss[-300:], infer_ya.loss[-300:]]
for i in range(2):
    axes[i].plot(loss[i])

axes[0].set_title('ELBO OA')
axes[1].set_title('ELBO YA')

#%% visualize posterior parameter estimates over subjects
g = sns.FacetGrid(pars_df_oa, col="parameter", hue='order', height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
plt.show()

g = sns.FacetGrid(pars_df_ya, col="parameter", hue='order', height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
plt.show()

#%% plot posterior distribution over groups
pars_df_oa['group'] = 'OA'
pars_df_ya['group'] = 'YA'

pars_df = pd.concat([pars_df_oa, pars_df_ya], ignore_index=True)

g = sns.FacetGrid(pars_df, col="parameter", hue='group', height=5, sharey=False, sharex=False, palette='Set1');
g = g.map(sns.kdeplot, 'value').add_legend();

# uncommment to save the output
# pars_df.to_csv('pars_post_samples.csv')
# g.fig.savefig('post_group_parameters.pdf', dpi=300)

#%%
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

    # get sample planning depth exceedance count of the first and second choice
    # exceedance count => number of times planning depth d had highest posterior probability
    exc_count = [np.array([np.sum(post_depth[t].argmax(-1) == i, 0) for i in range(3)]) for t in range(2)]
    
    return post_depth, m_prob, exc_count

post_depth_oa, m_prob_oa, exc_count_oa = get_posterior_stats(post_marg_oa)
np.savez('oa_plandepth_stats', post_depth_oa, m_prob_oa, exc_count_oa)

post_depth_ya, m_prob_ya, exc_count_ya = get_posterior_stats(post_marg_ya)
np.savez('ya_plandepth_stats', post_depth_ya, m_prob_ya, exc_count_ya)

