# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:30:41 2020

@author: Johannes
"""

import matplotlib.pyplot as plt
from os import getcwd, path
import pandas as pd
import seaborn as sns


# set data directory as relative path from this files directory
reppath = getcwd()
datadir = path.join(reppath,"data")
    
datafile = path.join(datadir, "SAT_singleMiniblocks.csv")

grouplabels = ["young", "old"]
noiselabels = ["$\it{low~noise}$","$\it{high~noise}$"]
stepslabels = ["$\it{2~steps}$","$\it{3~steps}$"]
tasklabels  = ["$\it{SAT}$","$\it{SAW}$","$\it{SWM}$","$\it{IDP}$"]
paramlabels = ["$\it{alpha}$","$\it{beta}$","$\it{theta}$"]
groupcolors = ["dodgerblue","red"]
subjectplotcolor = "grey"

# help function to set outer lines of violinplot
def patch_violinplot(axis):
     from matplotlib.collections import PolyCollection
     ax = axis
     isya = True
     for art in ax.get_children():
          if isinstance(art, PolyCollection):
              if isya:
                  art.set_edgecolor(groupcolors[0])
                  isya = False
              else:
                  art.set_edgecolor(groupcolors[1])
                  isya = True


#%% import data
df = pd.read_csv(datafile, index_col=0, dtype={'ID':object})

# scale conditions as binary indicators
df["noise"]=df["noise"] - df["noise"].min()
df["steps"]=df["steps"] - df["steps"].min()


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
dfsubs = df.groupby('ID').mean().reset_index()
dfsubsNoise = df.groupby(['ID','noise']).mean().reset_index()
dfsubsSteps = df.groupby(['ID','steps']).mean().reset_index()
subsCount = len(dfsubs)
subsIDs = pd.unique(dfsubs.ID)

# plot row1
#------------------------

ycol = 'MeanPD'
ylabel = 'Mean Planning Depth'
ylims = [0.8,2.99]
xlabels = ['','',''] #['Age Group', 'Noise Level', 'Number of Steps']

sns.barplot(  ax=ax00, x="group", y=ycol, data=dfsubs, palette=groupcolors, ci=95, n_boot=10000, capsize=.2)
sns.stripplot(ax=ax00, x="group", y=ycol, data=dfsubs, color='black', jitter=0.1, alpha=0.3)


for t in range(subsCount):
    c = groupcolors[int(dfsubsNoise[dfsubsNoise.ID==subsIDs[t]].group.min())]
    sns.lineplot(ax=ax01, x="noise", y=ycol, data=dfsubsNoise[dfsubsNoise.ID==subsIDs[t]], color=c, linewidth=0.6, alpha=0.4)
    sns.stripplot(ax=ax01, x="noise",y=ycol, data=dfsubsNoise[dfsubsNoise.ID==subsIDs[t]], color=c, alpha=0.4, jitter=0, size=3)
    plt.setp(ax01.lines, zorder=1)
    plt.setp(ax01.collections, zorder=1)
sns.pointplot(ax=ax01, x="noise", y=ycol, data=dfsubsNoise, palette=groupcolors, hue="group", ci=95, n_boot=10000, capsize=.2, scale=1.5)
current_handles, current_labels = ax01.get_legend_handles_labels()
ax01.legend(current_handles, grouplabels, fontsize=legendsize, title_fontsize=legendsize, loc='upper left', title="Age Group")

  
for t in range(subsCount): 
    c = groupcolors[int(dfsubsSteps[dfsubsSteps.ID==subsIDs[t]].group.min())]
    sns.lineplot(ax=ax02, x="steps", y=ycol, data=dfsubsSteps[dfsubsSteps.ID==subsIDs[t]], color=c, linewidth=0.6, alpha=0.4)
    sns.stripplot(ax=ax02, x="steps",y=ycol, data=dfsubsSteps[dfsubsSteps.ID==subsIDs[t]], color=c, alpha=0.4, jitter=0, size=3)
    plt.setp(ax02.lines, zorder=1)
    plt.setp(ax02.collections, zorder=1)
sns.pointplot(ax=ax02, x="steps", y=ycol, data=dfsubsSteps, palette=groupcolors, hue="group", ci=95, n_boot=10000, capsize=.2, scale=1.5) 
ax02.get_legend().remove()


# adjust plots
ax00.set_ylabel(ylabel, fontsize=labelsize)
ax01.set_ylabel('')
ax02.set_ylabel('')
for t in range(3): 
    axs_top[t].set_xlabel(xlabels[t], fontsize=labelsize)
    axs_top[t].set_ylim(ylims[0],ylims[1])
ax00.set_xticks([],[])
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
ylims = [-25,135]

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['ID','group'], value_vars=datacols,
             var_name='Task', value_name='Performance')

sns.violinplot(ax=axbig1, x="Task", y="Performance", hue="group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axbig1)
axbig1.get_legend().remove()

axbig1.set_ylabel(ylabel, fontsize=labelsize) 
axbig1.set_xlabel(xlabel, fontsize=labelsize)
axbig1.tick_params(axis="x",direction="inout") 
axbig1.set_xticklabels(["","","",""])
axbig1.set_ylim(ylims[0],ylims[1])


# plot row3: plot for RT and Covariate RTs
#---------------------------------------------
xlabel = ""
covLegend = []
ylabel = 'RT (s)'
datacols0 = ['SAT_RT','SAW_RT']
datacols1 = ['SWM_RT','IDP_RT']
ylims0 = [-5,29]
ylims1 = [-0.5,7.]

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['ID','group'], value_vars=datacols0,
             var_name='Task', value_name='RT')
# plot
sns.violinplot(ax=axmed20, x="Task", y="RT", hue="group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axmed20)
axmed20.get_legend().remove()

# convert from wide to long format with Performance
dfsubsL = pd.melt(dfsubs, id_vars=['ID','group'], value_vars=datacols1,
             var_name='Task', value_name='RT')
# plot
sns.violinplot(ax=axmed21, x="Task", y="RT", hue="group", split=True, data=dfsubsL, palette=groupcolors)
patch_violinplot(axmed21)
axmed21.get_legend().remove()

axmed20.set_ylabel(ylabel, fontsize=labelsize) 
axmed21.set_ylabel("") 
axmed20.set_xlabel(xlabel, fontsize=labelsize)
axmed21.set_xlabel("")
axmed20.tick_params(axis="x",direction="inout", bottom=False, labelbottom=False, top=True, labeltop=True, pad = -22) 
axmed21.tick_params(axis="x",direction="inout", bottom=False, labelbottom=False, top=True, labeltop=True, pad = -22) 
axmed20.set_xticklabels(tasklabels[:2])
axmed21.set_xticklabels(tasklabels[2:])

axmed20.set_ylim(ylims0[0],ylims0[1])
axmed21.set_ylim(ylims1[0],ylims1[1])


# plot row4: model params (alpha, beta, theta)
#----------------------------------------------------------
ylabel = 'Value'
datacols = ['model_alpha','model_beta','model_theta']
cCount = len(datacols)

for i in range(cCount):
    # plot
    sns.violinplot(ax=axs_bottom[i], x="group", y=datacols[i], data=dfsubs, palette=groupcolors)
    patch_violinplot(axs_bottom[i])
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
