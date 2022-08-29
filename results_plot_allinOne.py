# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:30:41 2020

@author: Johannes
"""

import matplotlib.pyplot as plt
from os import name, getcwd
import pandas as pd
import seaborn as sns

# CONSTANTS
currentdir = getcwd()
if name == 'nt':
    #windows paths
    datadir = currentdir + '\\data\\'
if name == 'posix':
    #linux paths
    datadir = currentdir + '/data/'
    
datafile = datadir + "SAT_singleMiniblocks.xlsx"

grouplabels = ["young", "old"]
noiselabels = ["$\it{low~noise}$","$\it{high~noise}$"]
stepslabels = ["$\it{2~steps}$","$\it{3~steps}$"]
tasklabels  = ["$\it{SAT}$","$\it{SAW}$","$\it{SWM}$","$\it{IDP}$"]
paramlabels = ["$\it{alpha}$","$\it{beta}$","$\it{theta}$"]
groupcolors = ["dodgerblue","r"]
subjectplotcolor = "grey"

# help function to set outer lines of violinplot
def patch_violinplot(axis):
     from matplotlib.collections import PolyCollection
     ax = axis
     for art in ax.get_children():
          if isinstance(art, PolyCollection):
              art.set_edgecolor("white")


#%% import data
df = pd.read_excel(datafile)

# scale conditions as binary indicators
df["noise"]=df["noise"] - df["noise"].min()
df["steps"]=df["steps"] - df["steps"].min()

# # convert Covariate RTs from ms to seconds
# df['SAW_RT']=df['SAW_RT'] / 1000
# df['SWM_RT']=df['SWM_RT'] / 1000
# df['IDP_RT']=df['IDP_RT'] / 1000

# add colum for SAT performance (% of maximum possible points)
maxPoints = 2009   # maximum possible points gain from 10.000 depth3-agent simulations
df['SAT_PER'] = df['Total_points'] / maxPoints * 100


#%% plot main plot (MeanPD, Performance, RTs, Model Params)

# Shared Design Variables
labelsize  = 14
legendsize = 11

# Setup Figure
gs_main = plt.GridSpec(4,6, height_ratios=[2.5, 1, 1, 1],wspace=0,hspace=0)

fig = plt.figure(figsize=(10,12))
ax00    = fig.add_subplot(gs_main[0,0:2])
ax01    = fig.add_subplot(gs_main[0,2:4])
ax02    = fig.add_subplot(gs_main[0,4:6])
axs_top = [ax00,ax01,ax02]
axbig1  = fig.add_subplot(gs_main[1,:])
axmed20 = fig.add_subplot(gs_main[2,0:3])
axmed21 = fig.add_subplot(gs_main[2,3:6])
ax30    = fig.add_subplot(gs_main[3,0:2])
ax31    = fig.add_subplot(gs_main[3,2:4])
ax32    = fig.add_subplot(gs_main[3,4:6])
axs_bottom = [ax30, ax31, ax32]
legendHandles = []
legendTexts = []

for a in fig.axes:
    a.tick_params(        
    which='both',
    left=True,
    right=False,
    labelleft=True, 
    labelsize=labelsize)   

# calc subject-wise means 
dfsubs = df.groupby('Subject_ID').mean().reset_index()
dfsubsNoise = df.groupby(['Subject_ID','noise']).mean().reset_index()
dfsubsSteps = df.groupby(['Subject_ID','steps']).mean().reset_index()
subsCount = len(dfsubs)

# plot row1
#------------------------

ycol = 'MeanPD'
ylabel = 'Mean Planning Depth'
ylims = [0.8,3]
xlabels = ['','',''] #['Age Group', 'Noise Level', 'Number of Steps']

sns.barplot(  ax=ax00, x="Group", y=ycol, data=dfsubs, palette=groupcolors, ci=95, n_boot=10000, capsize=.2)
sns.stripplot(ax=ax00, x="Group", y=ycol, data=dfsubs, color='black', jitter=0.1, alpha=0.3)


for t in range(subsCount):
    c = groupcolors[int(dfsubsNoise[dfsubsNoise.Subject_Nr==t+1].Group.min())]
    sns.lineplot(ax=ax01, x="noise", y=ycol, data=dfsubsNoise[dfsubsNoise.Subject_Nr==t+1], color=c, linewidth=0.6, alpha=0.4)
    sns.stripplot(ax=ax01, x="noise",y=ycol, data=dfsubsNoise[dfsubsNoise.Subject_Nr==t+1], color=c, alpha=0.4, jitter=0, size=3)
    plt.setp(ax01.lines, zorder=1)
    plt.setp(ax01.collections, zorder=1)
sns.pointplot(ax=ax01, x="noise", y=ycol, data=dfsubsNoise, palette=groupcolors, hue="Group", ci=95, n_boot=10000, capsize=.2, scale=1.5)
ax01.get_legend().remove()

  
for t in range(subsCount): 
    c = groupcolors[int(dfsubsSteps[dfsubsSteps.Subject_Nr==t+1].Group.min())]
    sns.lineplot(ax=ax02, x="steps", y=ycol, data=dfsubsSteps[dfsubsSteps.Subject_Nr==t+1], color=c, linewidth=0.6, alpha=0.4)
    sns.stripplot(ax=ax02, x="steps",y=ycol, data=dfsubsSteps[dfsubsSteps.Subject_Nr==t+1], color=c, alpha=0.4, jitter=0, size=3)
    plt.setp(ax02.lines, zorder=1)
    plt.setp(ax02.collections, zorder=1)
sns.pointplot(ax=ax02, x="steps", y=ycol, data=dfsubsSteps, palette=groupcolors, hue="Group", ci=95, n_boot=10000, capsize=.2, scale=1.5) 
current_handles, current_labels = ax02.get_legend_handles_labels()
ax02.legend(current_handles, grouplabels, fontsize=legendsize, title_fontsize=legendsize, loc='upper left', title="Age Group")


# adjust plots
ax00.set_ylabel(ylabel, fontsize=labelsize)
ax01.set_ylabel('')
ax02.set_ylabel('')
for t in range(3): 
    axs_top[t].set_xlabel(xlabels[t], fontsize=labelsize)
    axs_top[t].set_ylim(ylims[0],ylims[1])
ax00.set_xticks([],[])
#ax00.set_xticklabels(["",""])
ax01.set_xticklabels(noiselabels)
ax01.tick_params(axis="x",direction="in",pad=-28)
ax02.set_xticklabels(stepslabels)
ax02.tick_params(axis="x",direction="in",pad=-28)
ax01.set_xlim(-0.4,1.4)
ax02.set_xlim(-0.4,1.4)
ax01.set_yticks([],[])
ax02.set_yticks([],[])


# plot row2: %-Performance in SAT (% of Max Points) and Covs (ACC: % correct)
#------------------------------------------------------------------------
xlabel = ""
covLegend = []
ylabel = 'Performance (%)'
datacols = ['SAT_PER','SAW_PER','SWM_PER','IDP_PER']
cCount = len(datacols)
ylims = [-25,125]

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['Subject_ID','Group'], value_vars=datacols,
             var_name='Task', value_name='Performance')

sns.violinplot(ax=axbig1, x="Task", y="Performance", hue="Group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axbig1)
axbig1.get_legend().remove()

axbig1.set_ylabel(ylabel, fontsize=labelsize) 
axbig1.set_xlabel(xlabel, fontsize=labelsize)
axbig1.tick_params(axis="x",direction="inout") 
axbig1.set_xticklabels(["","","",""])
axbig1.set_ylim(ylims[0],ylims[1])

#current_handles, current_labels = axbig1.get_legend_handles_labels()
#axbig1.legend(current_handles, grouplabels, fontsize=legendsize, title_fontsize=legendsize, title="Age Group")


# plot row3: plot for RT and Covariate RTs
#---------------------------------------------
xlabel = ""
covLegend = []
ylabel = 'RT (s)'
datacols0 = ['RT_1st','SAW_RT']
datacols1 = ['SWM_RT','IDP_RT']
ylims0 = [-2,27]
ylims1 = [-0.5,6.5]

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['Subject_ID','Group'], value_vars=datacols0,
             var_name='Task', value_name='RT')
# plot
sns.violinplot(ax=axmed20, x="Task", y="RT", hue="Group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axmed20)
axmed20.get_legend().remove()

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['Subject_ID','Group'], value_vars=datacols1,
             var_name='Task', value_name='RT')
# plot
sns.violinplot(ax=axmed21, x="Task", y="RT", hue="Group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axmed21)
axmed21.get_legend().remove()

axmed20.set_ylabel(ylabel, fontsize=labelsize) 
axmed21.set_ylabel("") 
#axmed21.tick_params(axis="y",direction="in", pad = -18)
axmed20.set_xlabel(xlabel, fontsize=labelsize)
axmed21.set_xlabel("")
axmed20.tick_params(axis="x",direction="inout", bottom=False, labelbottom=False, top=True, labeltop=True, pad = -22) 
axmed21.tick_params(axis="x",direction="inout", bottom=False, labelbottom=False, top=True, labeltop=True, pad = -22) 
axmed20.set_xticklabels(tasklabels[:2])
axmed21.set_xticklabels(tasklabels[2:])

axmed20.set_ylim(ylims0[0],ylims0[1])
axmed21.set_ylim(ylims1[0],ylims1[1])
#current_handles, current_labels = axmed20.get_legend_handles_labels()
#axmed20.legend(current_handles, grouplabels, fontsize=legendsize, title_fontsize=legendsize, title="Age Group")


# plot row4: model params (alpha, beta, theta)
#----------------------------------------------------------
ylabel = 'Value'
datacols = ['Model_alpha','Model_beta','Model_theta']
ylims = [[ -.07, .23],
         [-0.5 ,4.7 ],
         [-1.5 ,2.3 ]]
bws = [2.5, 0.5, 0.5]
cCount = len(datacols)

for i in range(cCount):
    # plot
    sns.violinplot(ax=axs_bottom[i], x="Group", y=datacols[i], data=dfsubs, palette=groupcolors, bw=bws[i])
    patch_violinplot(axs_bottom[i])
    axs_bottom[i].set_ylim(ylims[i][0],ylims[i][1])
    axs_bottom[i].set_ylabel('') 
    axs_bottom[i].set_xlabel('')
    axs_bottom[i].set_xticks([],[])
    axs_bottom[i].text(.5,.85,paramlabels[i],
                    horizontalalignment='center',
                    fontsize=labelsize,
                    transform=axs_bottom[i].transAxes)

axs_bottom[0].set_ylabel(ylabel, fontsize=labelsize)


plt.tight_layout()

fig.savefig('results.svg', format='svg')
fig.savefig('results.png', dpi=300)
