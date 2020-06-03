#!/usr/bin/env python
# coding: utf-8

# ## Evaluate the results
# 
# Based on: https://www.kaggle.com/nadjetba/text-to-meaning-with-multi-label-classification?scriptVersionId=12686831

# In[1]:


from evaluation import  softmax, normalize
from evaluation import ndcg_at_k, precision_at_k
from evaluation import EvaluationData, loadEvaluationData
from evaluation import getMethodName, makeMDRow, EnsembleConfig, TestConfig
from evaluation import findThreshold, getMetrics, makeEnsemble, testFunction, basicEvaluation

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report

import numpy as np
from copy import copy

import argparse

parser = argparse.ArgumentParser("Evaluate Results")
parser.add_argument("--LABEL_COL_NAME",default='Labels', help="Labels, Domain, MThesaurus, Topterm", type=str)
parser.add_argument("--title",default='Results Experiments', help="title-results", type=str)
parser.add_argument("--experiment",default='report', help="What is the name of the experiment? will be used as filename.", type=str)
parser.add_argument('--use_ensemble', default=False, help="If the data are augmented", action='store_true')
parser.add_argument('--use_dynamic_thresholding', default=False, action='store_true')

args = parser.parse_args()

LABEL_COL_NAME=args.LABEL_COL_NAME
use_ensemble=args.use_ensemble
use_dynamic_thresholding=args.use_dynamic_thresholding
# In[2]:


import pickle

with open('predictions/{}/validationDataOrg.pickle'.format(LABEL_COL_NAME), 'rb') as handle:
    validationDataOrg = pickle.load(handle)
with open('predictions/{}/testDataOrg.pickle'.format(LABEL_COL_NAME), 'rb') as handle:
    testDataOrg = pickle.load(handle)
with open('predictions/{}/validationDataExt.pickle'.format(LABEL_COL_NAME), 'rb') as handle:
    validationDataExt = pickle.load(handle)
with open('predictions/{}/testDataExt.pickle'.format(LABEL_COL_NAME), 'rb') as handle:
    testDataExt = pickle.load(handle)


    
# additional labels (zero-shot)
AdditionalColumnsLength=validationDataOrg.y_true.shape[1]-validationDataOrg.y_pred.shape[1]

validationDataOrg.y_pred=np.concatenate([validationDataOrg.y_pred, np.zeros((validationDataOrg.y_pred.shape[0],AdditionalColumnsLength))], axis=1)
testDataOrg.y_pred=np.concatenate([testDataOrg.y_pred, np.zeros((testDataOrg.y_pred.shape[0],AdditionalColumnsLength))], axis=1)
validationDataExt.y_pred=np.concatenate([validationDataExt.y_pred, np.zeros((validationDataExt.y_pred.shape[0],AdditionalColumnsLength))], axis=1)
testDataExt.y_pred=np.concatenate([testDataExt.y_pred, np.zeros((testDataExt.y_pred.shape[0],AdditionalColumnsLength))], axis=1)


# In[3]:


savename=args.experiment
saveFile=open('results/{}.md'.format(savename),'w')
saveFile.write('## '+args.title+'\n\n')

# In[4]:


method='- Basic - no Dynamic thresholding - no ensembling'
report, f1_val=basicEvaluation(validationDataOrg, testDataOrg, EnsembleConfig, plot=True, **TestConfig)
saveFile.write(method+"\n\n```\n"+report+'```\n\n')

globalBestF1, globalBestMethod = f1_val, method
print(report)

if use_dynamic_thresholding:

    # In[5]:


    from copy import copy
    dtConfig=copy(EnsembleConfig)
    dtConfig['dynamic_thresholding']=True


    # In[6]:
    for norm_type in ['max','std','none']:

        dtConfig['norm_type']=norm_type
        method=getMethodName(dtConfig)
        print(method)

        report, bestF1=testFunction(validationDataOrg, testDataOrg,
                 ensembleConfig=dtConfig, **TestConfig)

        saveFile.write(method+"\n\n"+report+'\n\n')

        if bestF1>globalBestF1:
            globalBestF1, globalBestMethod=bestF1, method
        print(report) # save to file




# ## Experiments on Ensemble
if use_ensemble and use_dynamic_thresholding: 
    # In[9]:


    ens_dt_Config=copy(EnsembleConfig)
    ens_dt_Config['dynamic_thresholding']=True
    ens_dt_Config['norm_type']='std'

    tst_Config=copy(TestConfig)
    tst_Config['minK']=8
    tst_Config['maxK']=13
    tst_Config['low_threshold']=0.2
    tst_Config['high_threshold']=0.5


    # In[10]:

    for etype in ['avg','min','max']:

        ens_dt_Config['etype']=etype
        method=getMethodName(ens_dt_Config)
        print(method)

        report, bestF1=testFunction( validationDataExt, testDataExt,
                     ensembleConfig=ens_dt_Config, **tst_Config)
        saveFile.write(method+"\n\n"+report+'\n\n')

        if bestF1>globalBestF1:
            globalBestF1, globalBestMethod=bestF1, method
        print(report) # save to file



if use_ensemble:
    ## NO DT


    # In[13]:


    from copy import copy
    ens_dt_Config=copy(EnsembleConfig)
    ens_dt_Config['dynamic_thresholding']=False
    ens_dt_Config['norm_type']='std'

    tst_Config=copy(TestConfig)
    tst_Config['minK']=1
    tst_Config['maxK']=2
    tst_Config['low_threshold']=0.2
    tst_Config['high_threshold']=0.5


    # In[14]:

    for etype in ['avg','min','max']:
        ens_dt_Config['etype']=etype
        method=getMethodName(ens_dt_Config)
        print(method)

        report, bestF1=testFunction( validationDataExt, testDataExt,
                     ensembleConfig=ens_dt_Config, **tst_Config)
        saveFile.write(method+"\n\n"+report+'\n\n')

        if bestF1>globalBestF1:
            globalBestF1, globalBestMethod=bestF1, method
        print(report) # save to file



# ## IR Metrics

# In[5]:


method='- Precision@k'
report=''
for k in range(1,20):
    report+='precision@{} = {}'.format(k,precision_at_k(testDataOrg.y_true, testDataOrg.y_pred,k))+'\n'
    
saveFile.write(method+"\n\n```\n"+report+'```\n\n')
print(report)


# In[6]:


method='- nDCG@k'
report=''
for k in range(1,20):
    report+='nDCG@{} = {}'.format(k,ndcg_at_k(testDataOrg.y_true, testDataOrg.y_pred,k))+'\n'
    
saveFile.write(method+"\n\n```\n"+report+'```\n\n')
print(report)


# In[7]:


saveFile.write('-'*10+'\n\n'+'Best method: '+globalBestMethod)
saveFile.close()