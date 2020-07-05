import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import random

## Constants for load dataset
TEXT_FIELD = 'text'
SPLIT_FIELD = 'split'
TRAIN_LABEL = 'train'
VALIDATION_LABEL = 'val'
TEST_LABEL = 'test'
NO_SPLIT_LABEL = 'no split'

FILE_ID_FIELD = 'celex_id'
TRAIN_FILENAME = 'train.txt'
VALIDATION_FILENAME = 'validation.txt'
TEST_FILENAME = 'test.txt'
LANG_FIELD = 'lang'

## Constants for plotting results
MAX_K = 19
CYCLE_LABEL = 'Cycle'
Y_LABEL = 'score'
F1_SCORE_LABEL = 'F1-score'
PRAT_LABEL = 'RP@'
NDCGAT_LABEL = 'nDCG@'
SELECTED_K = [1, 3, 5, 10]
RESULTS_FILENAME = './results.csv'
F1_SCORE_FIGURE_SAVEFILE = 'F1-score.png'
PRATK_FIGURE_SAVEFILE = 'RP@k'
NDCGATK_FIGURE_SAVEFILE = 'nDCG@k'


## Plot results
def plotFigures(data, y, savepath):
    fig = data.plot(x=CYCLE_LABEL, y=y)
    fig.set_ylabel(Y_LABEL)
    fig.grid()
    plt.savefig(savepath)


def plotResults(saveDir, filepath):
    Columns = [F1_SCORE_LABEL] \
              + [PRAT_LABEL + str(k + 1) for k in range(MAX_K)] \
              + [NDCGAT_LABEL + str(k + 1) for k in range(MAX_K)]

    data = pd.read_csv(filepath, names=Columns)
    data[CYCLE_LABEL] = [str(k + 1) for k in range(len(data))]
    data.set_index(CYCLE_LABEL)

    savepath = saveDir + "/" + F1_SCORE_FIGURE_SAVEFILE
    y = [F1_SCORE_LABEL]
    plotFigures(data, y, savepath)

    savepath = saveDir + "/" + PRATK_FIGURE_SAVEFILE
    y = [PRAT_LABEL + str(k) for k in SELECTED_K]
    plotFigures(data, y, savepath)

    savepath = saveDir + "/" + NDCGATK_FIGURE_SAVEFILE
    y = [NDCGAT_LABEL + str(k) for k in SELECTED_K]
    plotFigures(data, y, savepath)


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def performFinetuningCycle(learner, unfreeze_to, n_iterations, max_lr, model_name, continue_from, seed=42):
    seed_all(seed)
    learner.load("{}/{}".format(model_name, continue_from - 1))
    if unfreeze_to == -100:
        learner.unfreeze()
    elif unfreeze_to < 0:
        learner.freeze_to(unfreeze_to)
    learner.fit_one_cycle(n_iterations, max_lr=max_lr, moms=(0.8, 0.7))
    learner.save("{}/{}".format(model_name, continue_from))

    return learner


def lrFind(learner, unfreeze_to, n_iterations, max_lr, model_name, continue_from, seed=42,
           figname="lrFind/default.png"):
    seed_all(seed)
    learner.load("{}/{}".format(model_name, continue_from - 1))
    if unfreeze_to < 0:
        learner.freeze_to(unfreeze_to)
    learner.lr_find()

    plt.figure()
    learner.recorder.plot(skip_end=10, suggestion=True)
    plt.savefig(figname)


def prepareDataset(dataset_path, datasetSplit, uncased, trainLangs, testLangs):
    def getSplit(celex_id, lang, trainset, valset, testset):
        if celex_id in trainset:
            if lang not in trainLangs:
                return NO_SPLIT_LABEL
            return TRAIN_LABEL
        else:
            if lang not in testLangs:
                return NO_SPLIT_LABEL
            elif celex_id in valset:
                return VALIDATION_LABEL
            elif celex_id in testset:
                return TEST_LABEL
            else:
                return NO_SPLIT_LABEL
                # Load dataset

    data = pd.read_csv(dataset_path)
    if uncased:
        data[TEXT_FIELD] = data[TEXT_FIELD].apply(lambda x: x.lower())
    if len(datasetSplit) > 0:
        try:
            with open(datasetSplit + '/' + TRAIN_FILENAME) as fin:
                trainset = [line.strip() for line in fin]
            with open(datasetSplit + '/' + VALIDATION_FILENAME) as fin:
                valset = [line.strip() for line in fin]
            with open(datasetSplit + '/' + TEST_FILENAME) as fin:
                testset = [line.strip() for line in fin]
            data[SPLIT_FIELD] = data.apply(
                lambda w: getSplit(w[FILE_ID_FIELD], w[LANG_FIELD], trainset, valset, testset), axis=1)
        except Exception as ex:
            print(ex)
    return data
