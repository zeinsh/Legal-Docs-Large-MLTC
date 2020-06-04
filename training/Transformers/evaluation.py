from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from fastai.text import TextList, DatasetType
import matplotlib.pyplot as plt

import torch

def get_ground_truth(c2i, labels_col):
    num_labels=len(c2i)
    y_true=np.zeros((len(labels_col),num_labels))

    for i in range(len(labels_col)):
        labels=labels_col.iloc[i].split(';')
        for label in labels:
            y_true[i][c2i[label]]=1
    return y_true

def getPredictions(learn, test, vocab):
    test_data = TextList.from_df(test, cols='text', vocab=vocab)
    learn.data.add_test(test_data)
    y_pred, _ =  learn.get_preds(DatasetType.Test)
    y_pred = y_pred.numpy()
    return y_pred #np.concatenate([y_pred, np.zeros((y_pred.shape[0],89))], axis=1)



class EvaluationData:
    def __init__(self, y_pred, y_true, filenames, selected_group, columns):
        self.y_pred=y_pred
        self.y_true=y_true
        self.filenames=filenames
        self.selected_group=selected_group
        self.columns=columns

def loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, columns, split='val', original=True):
    assert split in ['val', 'test']
    evaluationData=df[df['split']==split]
#     if original:
#          evaluationData=evaluationData[evaluationData['original']==1]
    filenames=evaluationData['celex_id'].tolist()
    y_true_col=evaluationData[LABEL_COL_NAME]
    y_true=get_ground_truth(c2i, y_true_col)
    y_pred=getPredictions(learner, evaluationData, vocab)

    return EvaluationData(y_pred, y_true, filenames, selected_group, columns)


## Configurations 
def getMethodName(ensembleConfig):
    name='- '
    if ensembleConfig['dynamic_thresholding']:
        name+='Dynamic Thresholding ({}) | '.format(ensembleConfig['norm_type'])
    if ensembleConfig['etype']!='default':
        name+='Ensemble agg({})'.format(ensembleConfig['etype'])
    else:
        name+='No ensemble'
    return name

def makeMDRow(vec, boldmaxVal=False):
      if boldmaxVal==True:
        idx=1
        mx=vec[1]
        for i in range(2,len(vec)):
            if mx<vec[i]:
                mx,idx=vec[i],i
        vec[idx]='**{}**'.format(vec[idx])
        return '|'+'|'.join(vec)+'|'.replace(mx,'**'+mx+'**')
      else:
        return '|'+'|'.join(vec)+'|'

EnsembleConfig={
    'vote_threshold':0.2, 
    'etype':'default', 
    'dynamic_thresholding':False, 
    'norm_type':'max',
}

TestConfig={
    'minK':5,
    'maxK':15,    
    'low_threshold':0,
    'high_threshold':0.5,
}

## Helpers
def softmax(x, eps=0.0000000001):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0) +eps)

def normalize(arr, eps=0.0000000001, norm_type='max'):
  mn, mx=np.min(arr), np.max(arr)
  mean, std=np.mean(arr), np.std(arr)
  if norm_type=='none':
    return arr
  elif norm_type=='softmax':
    return softmax(np.array(arr),eps)
  elif norm_type=='max':
    return arr/(mx+eps)
  elif norm_type=='std':
    ret=(arr-mean)/std
    return normalize(ret, norm_type='min_max')
  else:
    return (arr-mn)/(mx-mn+eps)


def findThreshold(evaluationData, low=0, high=1, log=True):
    maxf1=-10
    best_threshold=0
    f1_scores=[]

    for thresh in np.arange(low,high,0.01):
        f1=f1_score(evaluationData.y_true, evaluationData.y_pred>thresh, average='micro')
        f1_scores.append(f1)
        if f1>maxf1:
            maxf1=f1
            best_threshold=thresh
        if log:
            print(thresh, f1)
    return best_threshold, maxf1, f1_scores

def multi_label_precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum()/(pred_pos.sum() + epsilon))#.item()
  
def multi_label_recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum()/(targs.sum() + epsilon))

def getMetrics(y_true, y_pred, threshold):
    f1   = f1_score(y_true, y_pred>threshold, average='micro')
    prec = precision_score(y_true, y_pred>threshold, average='micro')
    rec  = recall_score(y_true, y_pred>threshold, average='micro')
    return f1, prec, rec


def makeEnsemble(evaluationData, k=None,
                 vote_threshold=0.2, etype='default', dynamic_thresholding=False, norm_type='max'): # inference

    y_pred, y_true, filenames=np.copy(evaluationData.y_pred), np.copy(evaluationData.y_true), evaluationData.filenames
    
    unique_filenames=list(set(filenames))
    numberOfTests=len(unique_filenames)

    ret=np.zeros((numberOfTests ,y_true.shape[1]))  
    y_true2=np.zeros((numberOfTests ,y_true.shape[1]))
    
    
    if etype not in ['avg', 'max', 'min', 'vote']:
        etype='default'
        unique_filenames, ret, y_true2=filenames, y_pred, y_true
        numberOfTests=len(unique_filenames)
    if etype=='min':
        ret+=1000000

    numfound=0
    if etype!='default':
        for i in range(numberOfTests):
            last=-1
            while True:
                try:
                    last=filenames.index(unique_filenames[i],last+1)

                    if etype=='max':
                        ret[i,:]=np.maximum(ret[i,:],y_pred[last,:])
                    elif etype=='min':
                        ret[i,:]=np.minimum(ret[i,:],y_pred[last,:])
                    elif etype=='vote':
                        ret[i,:]+=(y_pred[last,:]>vote_threshold)
                    else:
                        ret[i,:]+=y_pred[last,:]
                    numfound+=1
                    y_true2[i,:]=y_true[last,:]
                except Exception as ex:
                    if  etype=='avg':
                        ret[i,:]/=numfound #avg
                    elif etype=='vote':
                        ret[i,:]=ret[i,:]>(numfound/2)
                    numfound=0
                    break

    if dynamic_thresholding:
        for i in range(numberOfTests):
            arrlast, arrsortlast=ret[i],ret[i].argsort()[::-1]
            arrlast[arrsortlast[k:]]=0
            arrlast[arrsortlast[:k]]=normalize(arrlast[arrsortlast[:k]], norm_type=norm_type)
            ret[i]=arrlast

    y_pred=ret#ret[:numberOfTests,evaluationData.selected_group]
    y_true=y_true2[:numberOfTests,evaluationData.selected_group]
    return EvaluationData(y_pred, y_true, unique_filenames,evaluationData.selected_group, evaluationData.columns)

def testFunction(validationData, testData, ensembleConfig={},minK=5, maxK=15, low_threshold=0, high_threshold=0.5):

    kvalues=['K']
    validation_f1_scores=['**Validation F1**']
    test_f1_scores=['**Test F1**']
    precisions=['**Precision**']
    recalls=['**Recall**']
    best_thresholds=['**Threshold**']
    bestF1=-1
    
    if minK is None:
        assert dynamic_thresholding==False
        searchRange=['-'] # no DT print - instead of k value 
    else:
        searchRange=range(minK, maxK)
        
    for k in searchRange:
        validEnsData=makeEnsemble(validationData, k=k, **ensembleConfig)
        testEnsData=makeEnsemble(testData, k=k, **ensembleConfig)
        best_threshold, f1_val, _=findThreshold(validEnsData, low=low_threshold, high=high_threshold, log=False)
        f1, prec, rec   = getMetrics(testEnsData.y_true, testEnsData.y_pred, best_threshold)
        kvalues.append(str(k))
        validation_f1_scores.append(str(np.round(f1_val,4)))
        test_f1_scores.append(str(np.round(f1,4)))
        best_thresholds.append(str(np.round(best_threshold,2)))
        precisions.append(str(np.round(prec,4)))
        recalls.append(str(np.round(rec,4)))

        bestF1=max(bestF1, f1_val)
        print("k=",k,";threshold",best_threshold, ";F1 (micro)", bestF1)
    
    report=''
    dashes=['---']*len(kvalues)
    
    report+=makeMDRow(kvalues)+'\n'
    report+=makeMDRow(dashes)+'\n'
    report+=makeMDRow(validation_f1_scores, boldmaxVal=True)+'\n'
    report+=makeMDRow(test_f1_scores)+'\n'
    report+=makeMDRow(precisions)+'\n'
    report+=makeMDRow(recalls)+'\n'
    report+=makeMDRow(best_thresholds)+'\n'
    return report, bestF1
    
def basicEvaluation(validationData, testData, ensembleConfig, low_threshold=0, high_threshold=0.5, minK=None, maxK=None, plot=False):
    validEnsData=makeEnsemble(validationData, **ensembleConfig)
    testEnsData=makeEnsemble(testData, dynamic_thresholding=False)

    best_threshold, maxf1, f1_scores=findThreshold(validEnsData, low=low_threshold, high=high_threshold, log=False)
    print('Best threshold is ', best_threshold,'; Best F1 score is', maxf1)

    f1_val=f1_score(testEnsData.y_true, testEnsData.y_pred>best_threshold, average='micro')
    f1, prec, rec   = getMetrics(testEnsData.y_true, testEnsData.y_pred, best_threshold)

    if plot:
        plt.title('F1 score / threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')

        plt.plot(np.arange(low_threshold,high_threshold,0.01), f1_scores)

    report=''
    report+='Threshold {}'.format(best_threshold)+'\n'
    report+="F1 (micro) {}".format(f1)+'\n'
    report+="P (micro) {}".format(prec)+'\n'
    report+="R (micro) {}".format(rec)+'\n'

    return report, f1_val


## IR Metrics
## https://gist.github.com/mblondel/7337391
import numpy as np


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)

def precision_at_k(y_true, y_score, k):
    sm=0
    for i in range(y_true.shape[0]):
        sm+=ranking_precision_score(y_true[i], y_score[i], k=k)
    return sm/y_true.shape[0]    


# https://gist.github.com/mblondel/7337391#file-letor_metrics-py-L236
import numpy as np
def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def ndcg_at_k(y_true, y_score, k, gains="exponential"):
    sm=0
    for i in range(y_true.shape[0]):
        sm+=ndcg_score(y_true[i], y_score[i], k=k, gains=gains)
    return sm/y_true.shape[0]

