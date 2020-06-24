import argparse

import pandas as pd
from pathlib import Path

SPLIT_LABEL='split'
TEXT_LABEL = 'text'
TRAIN_TAG='train'
CONCATENATATE_SYMBOL='\n'
OUTPUT_PATH = './tmp/'


TRAIN_OUTPUT_PATH = OUTPUT_PATH + 'train.txt'
TEST_OUTPUT_PATH = OUTPUT_PATH + 'test.txt'

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser("Prepare text data for LM finetuning")
    parser.add_argument("--dataset_path", help="path of the dataset", type=str)

    args = parser.parse_args()

    # args
    dataset_path =  args.dataset_path

    data=pd.read_csv(dataset_path)

    trainTexts = data[data[SPLIT_LABEL]==TRAIN_TAG][TEXT_LABEL]
    trainText = CONCATENATATE_SYMBOL.join(trainTexts)

    valTestTexts = data[~(data[SPLIT_LABEL]==TRAIN_TAG)][TEXT_LABEL]
    valTestText = CONCATENATATE_SYMBOL.join(valTestTexts)

    with open(TRAIN_OUTPUT_PATH, 'w') as fout:
        fout.write(trainText)

    with open(TEST_OUTPUT_PATH, 'w') as fout:
        fout.write(valTestText)
        
    