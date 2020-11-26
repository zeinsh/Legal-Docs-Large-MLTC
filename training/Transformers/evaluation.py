from sklearn.metrics import precision_score, recall_score, f1_score
from fastai.text import TextList, DatasetType
import matplotlib.pyplot as plt

import torch

VALIDATION_LABEL = 'val'
TEST_LABEL = 'test'

TestConfig = {
    'minK': 5,
    'maxK': 15,
    'low_threshold': 0,
    'high_threshold': 0.5,
}


class EvaluationData:
    def __init__(self, y_pred, y_true, filenames, selected_group, columns):
        self.y_pred = y_pred
        self.y_true = y_true
        self.filenames = filenames
        self.selected_group = selected_group
        self.columns = columns


def get_ground_truth(c2i, labels_col):
    num_labels = len(c2i)
    y_true = np.zeros((len(labels_col), num_labels))

    for i in range(len(labels_col)):
        labels = labels_col.iloc[i].split(';')
        for label in labels:
            y_true[i][c2i[label]] = 1
    return y_true


def getPredictions(learn, test, vocab):
    test_data = TextList.from_df(test, cols='text', vocab=vocab)
    learn.data.add_test(test_data)
    y_pred, _ = learn.get_preds(DatasetType.Test)
    y_pred = y_pred.numpy()
    return y_pred


def loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, columns, split='val'):
    assert split in ['val', 'test']
    evaluationData = df[df['split'] == split]
    filenames = evaluationData['celex_id'].tolist()
    y_true_col = evaluationData[LABEL_COL_NAME]
    y_true = get_ground_truth(c2i, y_true_col)
    y_pred = getPredictions(learner, evaluationData, vocab)

    # additional labels (zero-shot)
    AdditionalColumnsLength = y_true.shape[1] - y_pred.shape[1]
    y_pred = np.concatenate([y_pred, np.zeros((y_pred.shape[0], AdditionalColumnsLength))], axis=1)
    
    y_true = y_true[:,selected_group]
    y_pred = y_pred[:,selected_group]
    
    return EvaluationData(y_pred, y_true, filenames, selected_group, columns)


## Helpers
def findThreshold(evaluationData, low=0, high=1, log=True):
    maxf1 = -10
    best_threshold = 0
    f1_scores = []

    for thresh in np.arange(low, high, 0.01):
        f1 = f1_score(evaluationData.y_true, evaluationData.y_pred > thresh, average='micro')
        f1_scores.append(f1)
        if f1 > maxf1:
            maxf1 = f1
            best_threshold = thresh
        if log:
            print(thresh, f1)
    return best_threshold, maxf1, f1_scores


def multi_label_precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum() / (pred_pos.sum() + epsilon))  # .item()


def multi_label_recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum() / (targs.sum() + epsilon))


def getMetrics(y_true, y_pred, threshold):
    f1 = f1_score(y_true, y_pred > threshold, average='micro')
    prec = precision_score(y_true, y_pred > threshold, average='micro')
    rec = recall_score(y_true, y_pred > threshold, average='micro')
    return f1, prec, rec


def basicEvaluation(validationData, testData, low_threshold=0, high_threshold=0.5, minK=None, maxK=None, plot=False):
    best_threshold, maxf1, f1_scores = findThreshold(validationData, low=low_threshold, high=high_threshold, log=False)
    print('Best threshold is ', best_threshold, '; Best F1 score is', maxf1)

    f1_val = f1_score(testData.y_true, testData.y_pred > best_threshold, average='micro')

    if plot:
        plt.title('F1 score / threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')

        plt.plot(np.arange(low_threshold, high_threshold, 0.01), f1_scores)

    return f1_val


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
    if len(unique_y) == 2:
        pos_label = unique_y[1]
    else:
        pos_label = 1

    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def precision_at_k(y_true, y_score, k):
    sm = 0
    for i in range(y_true.shape[0]):
        sm += ranking_precision_score(y_true[i], y_score[i], k=k)
    return sm / y_true.shape[0]


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
    sm = 0
    for i in range(y_true.shape[0]):
        sm += ndcg_score(y_true[i], y_score[i], k=k, gains=gains)
    return sm / y_true.shape[0]


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
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
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    if len(unique_y) == 2:
        pos_label = unique_y[1]
    else:
        pos_label = 1
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in xrange(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in xrange(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


############### Evaluate model ##############
def performEvaluation(df, c2i, learner, vocab, LABEL_COL_NAME, COLUMNS, model_output_name, selected_group):
    _ = learner.load(model_output_name)

    validationDataOrg = loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS,
                                           split=VALIDATION_LABEL)
    testDataOrg = loadEvaluationData(df, c2i, learner, vocab, LABEL_COL_NAME, selected_group, COLUMNS, split=TEST_LABEL)

    f1_val = basicEvaluation(validationDataOrg, testDataOrg, plot=True, **TestConfig)

    ### make function for each loop
    prAtK = []
    for k in range(1, 20):
        prAtK.append(precision_at_k(testDataOrg.y_true, testDataOrg.y_pred, k))

    nDcgAtK = []
    for k in range(1, 20):
        nDcgAtK.append(ndcg_at_k(testDataOrg.y_true, testDataOrg.y_pred, k))

    return f1_val, prAtK, nDcgAtK
