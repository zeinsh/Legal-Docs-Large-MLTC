import json
import argparse
import pandas as pd
from pathlib import Path


def getData(dataPath, split):
    keys = ['celex_id', 'uri', 'type', 'Labels', 'title', 'header', 'recitals', 'main_body', 'attachments']
    data = {key: [] for key in keys + ['split', 'text']}
    for p in dataPath.glob('*.json'):
        datafile = json.loads(p.open().read())
        for key in keys:
            if key == 'main_body':
                main_body = '#NP#'.join(datafile.get(key, ['']))
                data[key].append(main_body)
            elif key == 'Labels':
                concepts = ';'.join(datafile.get('concepts', ['']))
                data[key].append(concepts)
            else:
                data[key].append(datafile.get(key, ''))
        text = '\n'.join([str(data[key][-1]) for key in \
                          ['title', 'recitals', 'attachments', 'header', 'main_body']])
        data['split'].append(split)
        data['text'].append(text)
    return pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare EurLex57K dataset")
    parser.add_argument("--save_path", help="path of the dataset", type=str)
    parser.add_argument("--dataset_path", help="path of original dataset in json format", type=str)

    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    OUTPUT_DIR = args.save_path

    path = Path(DATASET_PATH)
    trainData = getData(path / 'train', 'train')
    devData = getData(path / 'dev', 'val')
    testData = getData(path / 'test', 'test')

    pd.concat([trainData, devData, testData]).to_csv(OUTPUT_DIR, index=False)
