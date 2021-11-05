from transformersmd import MODEL_CLASSES, getTransformerProcecssor, getLearner

from helpers import prepareDataset
from helpers import performFinetuningCycle, lrFind
from helpers import plotResults

from evaluation import performEvaluation

from fastai.text import TextList
from pathlib import Path

import random
import argparse

########## CONSTANTS
TEXT_FIELD = 'text'
SPLIT_FIELD = 'split'
FILE_ID_FIELD = 'celex_id'
TRAIN_LABEL = 'train'
VALIDATION_LABEL = 'val'
TEST_LABEL = 'test'
NO_SPLIT_LABEL = 'no split'
LABEL_DELIM = ';'
COMMA = ','
EMPTY_STR = ''

def getByIndexOrLast(arr, index):
    if index < len(arr):
        return arr[index]
    else:
        return arr[-1]
    

def getSetOfLabels(data, labelColumn, split=None):
    if split==None:
        selected=data
    else:
        selected=data[data[SPLIT_FIELD]==split]
    return set(LABEL_DELIM.join(selected[labelColumn]).split(LABEL_DELIM))

def splitLabels(data, label, c2i):
    data=data[~data[label].isna()]

    # extract all labels
    labels_dict={}
    for i in range(len(data)):
        for tag in data[label].iloc[i].split(';'):
            labels_dict[tag] = labels_dict.get(tag,0)+1
    
    all_labels = [c2i[key] for key,value in labels_dict.items()]
    frequent_labels = [c2i[key] for key,value in labels_dict.items() if value>=50]
    low_frequent_labels = [c2i[key] for key,value in labels_dict.items() if value<50]

    return frequent_labels, low_frequent_labels, all_labels

def getLangResultsPath(lang, dataset_name, model_type, experiment_name, groupName):
    return "results/{}/{}/{}/{}/{}/".format(lang, dataset_name, model_type, experiment_name, groupName) + '/results.csv'

def getLangExperimentPath(lang, dataset_name, model_type, experiment_name, groupName):
    return "results/{}/{}/{}/{}/{}/".format(lang, dataset_name, model_type, experiment_name, groupName)

###################################### Prepare training settings #########
parser = argparse.ArgumentParser("Finetune transformer-based LM for multi-label classification")

## Data
parser.add_argument("--dataset_name", help="name of the dataset", type=str)
parser.add_argument("--dataset_path", help="path of the dataset", type=str)
parser.add_argument("--dataset_split_path", default="", help="path of the dataset", type=str)
parser.add_argument("--LABEL_COL_NAME", default='Labels', help="Labels, Domain, MThesaurus, Topterm", type=str)
parser.add_argument('--cased', default=0, help="set 1 if the model is cased", type=int)
parser.add_argument("--trainLanguages",
                    default="en",
                    help="Languages for training separated by commas: \n"
                         + "List of supported Languages: en, de, it, fr", type=str)
parser.add_argument("--testLanguages",
                    default="en",
                    help="Languages for validation/testing separated by commas: \n"
                         + "List of supported Languages: en, de, it, fr", type=str)

## Architecture
parser.add_argument("--model_type", default='bert', help="model type", type=str)
parser.add_argument("--pretrained_model_name", default='bert-base-uncased', help="model name or path", type=str)
parser.add_argument("--MAX_LEN", default=512, help="Max sequence len", type=int)

## Configuration
parser.add_argument("--BATCH_SIZE", default=4, help="Batch size", type=int)
parser.add_argument("--TOTAL_CYCLES", default=3, help="total number of cycles", type=int)
parser.add_argument("--START_CYCLE", default=1, help="Start/continue training from this cycle", type=int)
parser.add_argument("--N_ITERATIONS", default="12,12,12", help="number of iterations per cycle", type=str)
parser.add_argument("--LR", default="2e-04,5e-05,5e-06",
                    help="max learning rate for each cycle (last one will be default for the rest cycles)", type=str)
parser.add_argument("--UNFREEZED", default="-4,-8,-12",
                    help="unfreezed layers per cycle (last one will be default for the rest cycles)", type=str)

parser.add_argument("--experiment_name", help="name of the output model", type=str)
parser.add_argument('--lr_find', default=0, help="set to 1 to find learning-rate", type=int)

args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

## Data
dataset_name = args.dataset_name
dataset_path = args.dataset_path
dataset_split_path = args.dataset_split_path
LABEL_COL_NAME = args.LABEL_COL_NAME
uncased = (args.cased == 0)
testLangs = args.testLanguages.split(COMMA)
trainLangs = args.trainLanguages.split(COMMA)
trainLangsLabel="_".join(testLangs)
testLangsSet = set(trainLangs + testLangs + [trainLangsLabel])

## Architecture
model_type = args.model_type
pretrained_model_name = args.pretrained_model_name
MAX_LEN = args.MAX_LEN

## Configuration
bs = args.BATCH_SIZE
TOTAL_CYCLES = args.TOTAL_CYCLES
START_CYCLE = args.START_CYCLE
N_ITERATIONS = [int(value) for value in args.N_ITERATIONS.split(',')]
LR = [float(value) for value in args.LR.split(',')]
UNFREEZE = [int(value) for value in args.UNFREEZED.split(',')]

experiment_name = args.experiment_name
LR_FIND = (args.lr_find == 1)

# Parameters
seed = 42
use_fp16 = False
pad_first = bool(model_type in ['xlnet'])

assert LABEL_COL_NAME in ['MThesaurus', 'Domain', 'Topterm', 'Labels', 'ExtDesc', 'Domains', 'Descriptors']

## Create output dir for models
MODEL_PATH = "models/{}".format(experiment_name)
LR_PATH = "experiments/{}/{}/{}/lrFind/".format(dataset_name, model_type, experiment_name)
EXPERIMENT_PATH = "experiments/{}/{}/{}/".format(dataset_name, model_type, experiment_name)
RESULTS_SAVEPATH = EXPERIMENT_PATH + '/results.csv'
logfilename = EXPERIMENT_PATH + "/logs"

Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(LR_PATH).mkdir(parents=True, exist_ok=True)

for lang in testLangsSet:
    for groupName in ["allLabels", "frequent", "low_frequent"]:
        langExperimentPath = getLangExperimentPath(lang, dataset_name, model_type, experiment_name, groupName)
        Path(langExperimentPath).mkdir(parents=True, exist_ok=True)

################ Finetuning ################
# Load dataset
df = prepareDataset(dataset_path, dataset_split_path, uncased, trainLangs, testLangs)
df.fillna(EMPTY_STR, inplace=True)
train_idx = list(df[df[SPLIT_FIELD] == TRAIN_LABEL].index)
train_idx = [_idx for _idx in train_idx if df.iloc[_idx]['lang'] in trainLangs]
valid_idx = list(df[df[SPLIT_FIELD] == VALIDATION_LABEL].index)
random.shuffle(train_idx)
random.shuffle(valid_idx)

# Load Transformer model
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
transformer_processor = getTransformerProcecssor(tokenizer_class, pretrained_model_name, model_type, maxlen=MAX_LEN)
pad_idx = transformer_processor[1].vocab.tokenizer.pad_token_id

# prepare classification data
data_clas = (TextList.from_df(df, processor=transformer_processor, cols=TEXT_FIELD)
             .split_by_idxs(train_idx, valid_idx)
             .label_from_df(cols=LABEL_COL_NAME, label_delim=LABEL_DELIM)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

# Get classification learner
learner = getLearner(data_clas, pretrained_model_name, model_class, config_class, use_fp16, logfilename=logfilename,
                     append=True, model_type=model_type)
learner.save("{}/{}".format(experiment_name, 0))

# Task settings (I can get rid of them)
c2i = learner.data.c2i
COLUMNS = list(learner.data.classes)
vocab = learner.data.vocab

# Add labels which are not in train set
trainLabels=getSetOfLabels(df, LABEL_COL_NAME, TRAIN_LABEL)
allLabels=getSetOfLabels(df, LABEL_COL_NAME, None)
newLabels=allLabels-trainLabels
for singleLabel in newLabels:
    c2i[singleLabel] = len(c2i)
    
frequent_labels, low_frequent_labels, all_labels = splitLabels(df, LABEL_COL_NAME, c2i)
groupLabels={
    "allLabels":all_labels,
    "frequent":frequent_labels,
    "low_frequent":low_frequent_labels
}

for cycle in range(START_CYCLE, TOTAL_CYCLES + 1):
    max_lr = getByIndexOrLast(LR, cycle - 1)
    unfreeze_to = getByIndexOrLast(UNFREEZE, cycle - 1)
    n_iterations = getByIndexOrLast(N_ITERATIONS, cycle - 1)

    figname = "{}/{}.png".format(LR_PATH, cycle)
    lrFind(learner, unfreeze_to, n_iterations, max_lr, experiment_name, cycle, seed=seed, figname=figname)
    if not LR_FIND:
        learner = performFinetuningCycle(learner, unfreeze_to, n_iterations, max_lr, experiment_name, cycle, seed=seed)

    # Evaluation
    lastSavedModel = experiment_name + "/" + str(cycle)
    for groupName in groupLabels.keys():
        groupLabel = groupLabels[groupName]
        for lang in testLangsSet:
            langResutsPath=getLangResultsPath(lang, dataset_name, model_type, experiment_name, groupName)
            langExperimentPath = getLangExperimentPath(lang, dataset_name, model_type, experiment_name, groupName)

            # df lang
            if lang==trainLangsLabel:
                dfLang=df
            else:
                dfLang=df[df['lang']==lang]

            if (len(dfLang)==0):
                continue
                
            print(lang, len(dfLang), groupName, len(groupLabel)) 
            
            f1_val, prAtK, nDcgAtK = performEvaluation(dfLang, c2i, learner, vocab, LABEL_COL_NAME, COLUMNS, lastSavedModel, groupLabel)
            currentResults = [f1_val] + prAtK + nDcgAtK
            with open(langResutsPath, 'a') as fout:
                fout.write(COMMA.join([str(element) for element in currentResults]) + "\n")
            plotResults(langExperimentPath, langResutsPath)
