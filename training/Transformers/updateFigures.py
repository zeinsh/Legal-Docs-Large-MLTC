#!/usr/bin/env python
# coding: utf-8

from helpers import plotResults

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_K = 19
F1_SCORE_LABEL = 'F1-score'
PRAT_LABEL = 'RP@'
NDCGAT_LABEL = 'nDCG@'

EXPERIMENTS_PATH = Path('./experiments/')
RESULTS_FILENAME = 'results.csv'
COMPARISON_FIGURE_OUTPUT_NAME = "comparison.png"

# update results
CHOSEN_METRIC_LABELS = ['F1-score', 'RP@1', 'RP@3', 'RP@5', 'RP@10', 'RP@1', 'RP@3', 'RP@5', 'RP@10']
RESULTS_OUTPUT_METRICS_COL_NAMES = [F1_SCORE_LABEL] \
                                   + [PRAT_LABEL + str(k + 1) for k in range(MAX_K)] \
                                   + [NDCGAT_LABEL + str(k + 1) for k in range(MAX_K)]


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
    x = np.arange(len(CHOSEN_METRIC_LABELS))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 6))
    for _idx, key in enumerate(resultsDict.keys()):
        rect = ax.bar(x + _idx * width, resultsDict[key], width, label=key)
        autolabel(rect, ax)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(datasetName)
    ax.set_xticks(x)
    ax.set_xticklabels(CHOSEN_METRIC_LABELS)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    fig.tight_layout()

    plt.show()
    plt.savefig(savepath)


# update results
for datasetName in os.listdir(EXPERIMENTS_PATH):
    if ((EXPERIMENTS_PATH / datasetName).is_file()):
        continue
    for modelName in os.listdir(EXPERIMENTS_PATH / datasetName):
        if ((EXPERIMENTS_PATH / datasetName / modelName).is_file()):
            continue
        for experimentName in os.listdir(EXPERIMENTS_PATH / datasetName / modelName):
            if ((EXPERIMENTS_PATH / datasetName / modelName / experimentName).is_file()):
                continue
            experimentPath = EXPERIMENTS_PATH / datasetName / modelName / experimentName
            resultsPath = experimentPath / RESULTS_FILENAME
            if resultsPath.is_file():
                plotResults(str(experimentPath), str(resultsPath))

# In[15]:
for datasetName in os.listdir(EXPERIMENTS_PATH):
    if ((EXPERIMENTS_PATH / datasetName).is_file()):
        continue
    allResults = {}
    for modelName in os.listdir(EXPERIMENTS_PATH / datasetName):
        if ((EXPERIMENTS_PATH / datasetName / modelName).is_file()):
            continue
        for experimentName in os.listdir(EXPERIMENTS_PATH / datasetName / modelName):
            if ((EXPERIMENTS_PATH / datasetName / modelName / experimentName).is_file()):
                continue
            experimentPath = EXPERIMENTS_PATH / datasetName / modelName / experimentName
            resultsPath = experimentPath / RESULTS_FILENAME
            if resultsPath.is_file():
                data = pd.read_csv(resultsPath, names=RESULTS_OUTPUT_METRICS_COL_NAMES)
                singleResult = list(data.max().round(3)[CHOSEN_METRIC_LABELS])
                allResults[modelName + "/" + experimentName] = singleResult
    savepath = EXPERIMENTS_PATH / datasetName / COMPARISON_FIGURE_OUTPUT_NAME
    plotBars(allResults, datasetName, savepath)