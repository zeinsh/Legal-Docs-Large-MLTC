import pandas as pd
import numpy as np
from pathlib import Path
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
import argparse
# !pip install scikit-multilearn

## check train-set labels
def getLabelDicts(data):
    alllabels = []

    for i in range(len(data)):
        alllabels += data[LABEL].iloc[i].split(';')
    alllabels = list(set(alllabels))
    print("Total number of labels", len(alllabels))

    label2id = {alllabels[i]: i for i in range(len(alllabels))}
    id2label = {v: k for k, v in label2id.items()}

    return label2id, id2label

def convert_lst_to_int(lst, label2id):
    return [label2id.get(x, -1) for x in lst]

def getSplit(idx, ids_train, ids_val, ids_test):
        if idx in ids_train:
            return 'train'
        elif idx in idx_val:
            return 'val'
        elif idx in idx_test:
            return 'test'
        else:
            return 'none'

if __name__=="__main__":
    parser = argparse.ArgumentParser("Iterative split")
    parser.add_argument("--OUTPUT_DIR", help="directory to save the splits", type=str)
    parser.add_argument("--DATASET_PATH", help="path to the dataset", type=str)
    parser.add_argument("--LABEL", default="Descriptors",
                        help="Label type: Descriptors, Topterm, ExtDesc, MThesaurus, Domains", type=str)
    args = parser.parse_args()

    OUTPUT_DIR = args.OUTPUT_DIR
    DATASET_PATH = args.DATASET_PATH
    LABEL = args.LABEL

    print("Loading data ...")
    data = pd.read_csv(DATASET_PATH)
    data = data[~data[LABEL].isna()]
    assert LABEL in data.columns

    print("Prepare labels' dictionary ...")
    label2id, id2label = getLabelDicts(data)

    print("Prepare data before splitting ...")
    idx=np.array([i for i in range(len(data))])
    idx=idx.reshape(len(idx),1)
    y=[
        convert_lst_to_int(data[LABEL].iloc[i].split(';'), label2id)
        for i in range(len(data))
    ]
    ln=max([len(yi) for yi in y])
    Y=np.array([yi+[-1]*(ln-len(yi)) for yi in y])  # make 2D array

    print("Getting training ids, it might take a long time")
    idx_train, _, idx_val_test, y_val_test = iterative_train_test_split(idx, Y, test_size=0.2)  # about 30 minutes

    print("Getting validation, test ids")
    idx_val, _, idx_test, _ = iterative_train_test_split(idx_val_test, y_val_test, test_size=0.5)

    print("Make sure that all labels in test, validation splits are in training splits ... ")
    alllabels = []
    for i in range(len(idx_train)):
        alllabels += data['Descriptors'].iloc[idx_train[i][0]].split(';')
    alllabels = set(alllabels)

    trainList = idx_train.reshape(len(idx_train)).tolist()
    valList = idx_val.reshape(len(idx_val)).tolist()
    testList = idx_test.reshape(len(idx_test)).tolist()
    data['split'] = np.arange(len(data))
    data['split'] = data['split'].apply(lambda w: getSplit(w, trainList, valList, testList))

    for i in range(len(data)):
        if data['split'].iloc[i] == 'train':
            continue
        isTrain = False
        labels = data[LABEL].iloc[i].split(';')
        for label in labels:
            if label not in alllabels:
                alllabels.add(label)
                isTrain = True
        if isTrain:
            data['split'].iloc[i] = 'train'

    print("Save splits to output directory")
    filenames_train_set = data[data['split'] == 'train']['celex_id'].to_list()
    filenames_val_set = data[data['split'] == 'val']['celex_id'].to_list()
    filenames_test_set = data[data['split'] == 'test']['celex_id'].to_list()

    outputPath = Path(OUTPUT_DIR)
    outputPath.mkdir(parents=True, exist_ok=True)
    with open(outputPath / 'train.txt', 'w') as fout:
        fout.write('\n'.join(filenames_train_set))
    with open(outputPath / 'val.txt', 'w') as fout:
        fout.write('\n'.join(filenames_val_set))
    with open(outputPath / 'test.txt', 'w') as fout:
        fout.write('\n'.join(filenames_test_set))