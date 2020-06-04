#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 

import os

import torch
import torch.optim as optim

import random 

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *


# In[2]:

import argparse

parser = argparse.ArgumentParser("Finetune bert for multi-label classification")
parser.add_argument("--LABEL_COL_NAME",default='Labels', help="Labels, Domain, MThesaurus, Topterm", type=str)
parser.add_argument("--TOTAL_CYCLES",default=3, help="total number of cycles", type=int)
parser.add_argument("--N_ITERATIONS",default="12,12,12", help="number of iterations per cycle", type=str)
parser.add_argument("--LR",default="2e-04,5e-05,5e-06", help="max learning rate for each cycle (last one will be default for the rest cycles)", type=str)
parser.add_argument("--UNFREEZED",default="-4,-8,-12", help="unfreezed layers per cycle (last one will be default for the rest cycles)", type=str)
parser.add_argument("--model_output_name", help="name of the output model", type=str)
parser.add_argument("--model_type",default='bert', help="model type", type=str)
parser.add_argument("--pretrained_model_name",default='bert-base-uncased', help="model name or path", type=str)
parser.add_argument("--MAX_LEN",default=512, help="Max sequence len", type=int)
parser.add_argument("--dataset_path", help="path of the dataset", type=str)
parser.add_argument('--cased', default=False, help="cased model", action='store_true')
parser.add_argument('--continue_training', default=False, help="continue training", action='store_true')
parser.add_argument('--use_dynamic_thresholding', default=False, action='store_true')

args = parser.parse_args()


# Inputs
LABEL_COL_NAME=args.LABEL_COL_NAME
model_output_name=args.model_output_name
logfilename="logs/"+model_output_name
dataset_path=args.dataset_path
pretrained_model_name = args.pretrained_model_name
uncased=not args.cased
continue_training=args.continue_training
model_type = args.model_type
MAX_LEN=args.MAX_LEN

# N_ITERATIONS=args.N_ITERATIONS
TOTAL_CYCLES=args.TOTAL_CYCLES
LR=[float(value) for value in args.LR.split(',')]
UNFREEZE=[int(value) for value in args.UNFREEZED.split(',')]
N_ITERATIONS=[int(value) for value in args.N_ITERATIONS.split(',')]

## print review
print("LABEL_COL_NAME",LABEL_COL_NAME)
print("model_output_name",model_output_name)
print("dataset_path",dataset_path)
print("pretrained_model_name",pretrained_model_name)
print("uncased",uncased)
print("Continue training", continue_training)
print("model_type",model_type)
print("TOTAL_CYCLES",TOTAL_CYCLES)
print("N_ITERATIONS",N_ITERATIONS)
print("LR",LR)
print("UNFREEZE",UNFREEZE)
print("MAX_LEN", MAX_LEN)

# Parameters
bs = 4
seed = 42
use_fp16 = False
pad_first = bool(model_type in ['xlnet'])

assert LABEL_COL_NAME in ['MThesaurus','Domains','Topterm','Labels','ExtDesc','Domain']


# In[3]:


df=pd.read_csv(dataset_path)
if uncased:
    df['text']=df['text'].apply(lambda x:x.lower())

assert LABEL_COL_NAME in df.columns    


# In[4]:


from BertTransformer import MODEL_CLASSES, getTransformerProcecssor, CustomTransformerModel, getListLayersBert, getLearner
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
transformer_processor = getTransformerProcecssor(tokenizer_class, pretrained_model_name, model_type, maxlen=MAX_LEN)
pad_idx = transformer_processor[1].vocab.tokenizer.pad_token_id


# In[5]:

df['filename']=df['celex_id']
import numpy as np
print(df.columns)
train_idx=list(df[df['split']=='train'].index)  
valid_idx=list(df[df['split']=='val'].index)

data_clas = (TextList.from_df(df, processor=transformer_processor, cols='text')
         .split_by_idxs(train_idx, valid_idx)
         .label_from_df(cols=LABEL_COL_NAME, label_delim=';')
         .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

learner=getLearner(data_clas, pretrained_model_name, model_class, config_class, use_fp16, logfilename=logfilename, append=True,     model_type=model_type)
print('done')



# ## Extract results

# In[7]:


from evaluation import get_ground_truth, getPredictions, softmax, normalize
from evaluation import ndcg_at_k, precision_at_k, EvaluationData, loadEvaluationData

# Task settings
c2i=learner.data.c2i
COLUMNS=list(learner.data.classes)

## update c2i
print("len c2i before", len(c2i))
for i in range(len(df)):
    labels_raw=df[LABEL_COL_NAME].iloc[i]
    for singlelabel in labels_raw.split(';'):
        if singlelabel not in c2i.keys():
            c2i[singlelabel]=len(c2i)
            COLUMNS.append(singlelabel)
print("len c2i after", len(c2i))


vocab=learner.data.vocab

if 'original' not in df.columns:
    df['original']=1


# In[8]:


_=learner.load(model_output_name)


# In[9]:


descriptors_ids=[learner.data.c2i[k] for k in learner.data.c2i.keys()  if not (k.startswith('Do') or k.startswith('MT'))]
domains_ids=[learner.data.c2i[k] for k in learner.data.c2i.keys()  if k.startswith('Do')]
mthesaurus_ids=[learner.data.c2i[k] for k in learner.data.c2i.keys()  if k.startswith('MT')]
selected_group=descriptors_ids


# In[10]: 


validationDataOrg=loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS, split='val', original=True)
testDataOrg=loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS, split='test', original=True)


# In[11]:


validationDataExt=loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS, split='val', original=False)
testDataExt=loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS, split='test', original=False)

# In[12]:


import pickle

if not os.path.exists('predictions/{}'.format(LABEL_COL_NAME)):
    os.mkdir('predictions/{}'.format(LABEL_COL_NAME))

# save
with open('predictions/{}/validationDataOrg.pickle'.format(LABEL_COL_NAME), 'wb') as handle:
    pickle.dump(validationDataOrg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('predictions/{}/testDataOrg.pickle'.format(LABEL_COL_NAME), 'wb') as handle:
    pickle.dump(testDataOrg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('predictions/{}/validationDataExt.pickle'.format(LABEL_COL_NAME), 'wb') as handle:
    pickle.dump(validationDataExt, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('predictions/{}/testDataExt.pickle'.format(LABEL_COL_NAME), 'wb') as handle:
    pickle.dump(testDataExt, handle, protocol=pickle.HIGHEST_PROTOCOL)

