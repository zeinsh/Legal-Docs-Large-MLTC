import argparse

import numpy as np
import pandas as pd
from pathlib import Path

SPLIT_LABEL='split'
TEXT_LABEL = 'text'
TRAIN_TAG='train'
VAL_TAG='val'
TEST_TAG='test'
CONCATENATATE_SYMBOL='\n'

if __name__=="__main__":
    parser = argparse.ArgumentParser("Prepare text data for LM finetuning")
    parser.add_argument("--dataset_path", help="path of the dataset", type=str)
    parser.add_argument("--train_output_path", help="temporary text file to save training texts", type=str)
    parser.add_argument("--test_output_path", help="temporary text file to save test texts", type=str)

    args = parser.parse_args()

    # args
    dataset_path =  args.dataset_path
    train_output_path = args.train_output_path
    test_output_path = args.test_output_path
    
    # prepare directories
    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output_path).parent.mkdir(parents=True, exist_ok=True)

    # load data
    data=pd.read_csv(dataset_path)

    # get train texts
    trainTexts = data[(~(data[SPLIT_LABEL]==VAL_TAG) & (~(data[SPLIT_LABEL]==TEST_TAG)))][TEXT_LABEL]
    trainTexts = trainTexts.iloc[np.random.permutation(len(trainTexts))]
    trainText = CONCATENATATE_SYMBOL.join(trainTexts)
    
    # get validattion&test texts
    valTestTexts = data[((data[SPLIT_LABEL]==VAL_TAG) | (data[SPLIT_LABEL]==TEST_TAG))][TEXT_LABEL]
    valTestTexts = valTestTexts.iloc[np.random.permutation(len(valTestTexts))]
    valTestText = CONCATENATATE_SYMBOL.join(valTestTexts)
    
    print("NUMBER OF TRAINING TEXTS :",len(trainTexts))
    print("NUMBER OF VAL/TEST TEXTS:", len(valTestTexts))
    
    # save texts
    with open(train_output_path, 'w') as fout:
        fout.write(trainText)

    with open(test_output_path, 'w') as fout:
        fout.write(valTestText)