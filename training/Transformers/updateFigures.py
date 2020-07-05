#!/usr/bin/env python
# coding: utf-8

# In[14]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plotBars(resultsDict, datasetName, savepath):
    x = np.arange(len(LABELS))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots(figsize=(15,6))
    for _idx, key in enumerate(resultsDict.keys()):
        rect = ax.bar(x + _idx * width, resultsDict[key], width, label=key)
        autolabel(rect, ax)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(datasetName)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    fig.tight_layout()

    plt.show()
    plt.savefig(savepath)


# In[5]:


from pathlib import Path
import os
from helpers import plotResults
import pandas as pd
EXPERIMENTS_PATH


# In[9]:


# update results
EXPERIMENTS_PATH = Path('./experiments/')
for datasetName in os.listdir(EXPERIMENTS_PATH):
    if ((EXPERIMENTS_PATH/datasetName).is_file()):
        continue
    for modelName in os.listdir(EXPERIMENTS_PATH/datasetName):
        if ((EXPERIMENTS_PATH/datasetName/modelName).is_file()):
            continue
        for experimentName in os.listdir(EXPERIMENTS_PATH/datasetName/modelName):
            if ((EXPERIMENTS_PATH/datasetName/modelName/experimentName).is_file()):
                continue
            experimentPath=EXPERIMENTS_PATH/datasetName/modelName/experimentName
            resultsPath = experimentPath/'results.csv'
            if resultsPath.is_file():
                plotResults(str(experimentPath), str(resultsPath))


# In[15]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# update results
MAX_K=19
F1_SCORE_LABEL='F1-score'
PRAT_LABEL = 'RP@'
NDCGAT_LABEL = 'nDCG@'

Columns = [F1_SCORE_LABEL]         +[PRAT_LABEL+str(k+1) for k in range(MAX_K)]         +[NDCGAT_LABEL+str(k+1) for k in range(MAX_K)]

LABELS = ['F1-score', 'RP@1', 'RP@3', 'RP@5', 'RP@10', 'RP@1', 'RP@3', 'RP@5', 'RP@10']

    
EXPERIMENTS_PATH = Path('./experiments/')
for datasetName in os.listdir(EXPERIMENTS_PATH):
    if ((EXPERIMENTS_PATH/datasetName).is_file()):
        continue
    allResults = {}
    for modelName in os.listdir(EXPERIMENTS_PATH/datasetName):
        if ((EXPERIMENTS_PATH/datasetName/modelName).is_file()):
            continue
        for experimentName in os.listdir(EXPERIMENTS_PATH/datasetName/modelName):
            if ((EXPERIMENTS_PATH/datasetName/modelName/experimentName).is_file()):
                continue
            experimentPath=EXPERIMENTS_PATH/datasetName/modelName/experimentName
            resultsPath = experimentPath/'results.csv'
            if resultsPath.is_file():
                data=pd.read_csv(resultsPath, names=Columns)
                singleResult = list(data.max().round(3)[LABELS])
                allResults[modelName+"/"+experimentName] = singleResult
    savepath=EXPERIMENTS_PATH/datasetName/"comparison.png"
    plotBars(allResults, datasetName, savepath)


# In[12]:


data.max().round(3)

