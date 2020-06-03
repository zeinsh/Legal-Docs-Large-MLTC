import os
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from EuroVocAnalyzeTool import Graph, EuroVocAnalyzeTool
import pickle
import argparse

LANGUAGES = "en"
SAVE_PATH = "datasets/jrc_en_basic.csv"


## Micro-Thesaurus
def get_MThesaurus(descriptors, analyzeTool):
    ret = []
    for desc_id in descriptors.split(';'):
        mthes = analyzeTool.getThesaurusByDescId(desc_id)
        if mthes:
            for mthes_id in mthes:
                if not mthes_id in ret:
                    ret.append(mthes_id)
    return ';'.join(ret)


## Domains
def get_domains(descriptors, analyzeTool):
    ret = []
    for desc_id in descriptors.split(';'):
        domains = analyzeTool.getDomainsByDescId(desc_id)
        if domains:
            for domain_id in domains:
                if not domain_id in ret:
                    ret.append(domain_id)
    return ';'.join(ret)


## Topterms
def get_topterms(descriptors, analyzeTool):
    ret = []
    for desc_id in descriptors.split(';'):
        topterms = analyzeTool.getTopTermsByDescid(desc_id)
        if topterms:
            for topterm_id in topterms:
                if not topterm_id in ret:
                    ret.append(topterm_id)
        else:
            ret.append(desc_id)
    return ';'.join(ret)


## extended Descriptors
def get_extDesc(descriptors, analyzeTool):
    ret = []
    for desc_id in descriptors.split(';'):
        topterms = analyzeTool.getParents(desc_id)
        if topterms:
            for topterm_id in topterms:
                if not topterm_id in ret:
                    ret.append(topterm_id)
        else:
            ret.append(desc_id)
    return ';'.join(ret)


def parseXML(path, filename, sep='\n', section_sep=' #S# '):
    with open(path + '/' + filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    body = signature = annex = ''

    try:
        tagclasscode = ';'.join([p.text for p in parsedXml.find('textclass').find_all('classcode')])
    except:
        tagclasscode = ''

    divs = parsedXml.find('text').find_all('div')
    for div in divs:
        text = sep.join([p.text for p in div.find_all('p')])
        if div.get('type') == 'body':
            body = text
        elif div.get('type') == 'signature':
            signature = text
        elif div.get('type') == 'annex':
            annex = text
    return body + section_sep + signature + section_sep + annex, tagclasscode


def prepareDataset(languages, datasetSplit, save_path):
    def getSplit(celex_id, trainset, valset, testset):
        if celex_id in trainset:
            return 'train'
        elif celex_id in valset:
            return 'val'
        elif celex_id in testset:
            return 'test'
        else:
            return 'no split'

    dir_path = SAVE_PATH[:save_path.rfind('/') + 1]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    COLNAMES = ['celex_id', 'lang', 'year', 'text', 'Descriptors']
    data = pd.DataFrame(columns=COLNAMES)

    for lang in languages.split(","):
        for dirname in tqdm(os.listdir("./tmp/" + lang)):
            path = "./tmp/" + lang + "/" + dirname
            if os.path.isdir(path):
                for filename in os.listdir(path):
                    celex_id = filename[3:-7]
                    try:
                        text, tagclasscode = parseXML(path, filename, sep=' #NP# ')
                        data.loc[len(data)] = [celex_id, lang, dirname, text, tagclasscode]
                    except Exception as ex:
                        print(ex)
                        print(path + '/' + filename)

    with open('data/EuroVocAnalysisTool.pickle', 'rb') as handle:
        analyzeTool = pickle.load(handle)

    # Extend Dataset
    data['Domains'] = data['Descriptors'].apply(lambda w: get_domains(w, analyzeTool))
    data['MThesaurus'] = data['Descriptors'].apply(lambda w: get_MThesaurus(w, analyzeTool))
    data['Topterm'] = data['Descriptors'].apply(lambda w: get_topterms(w, analyzeTool))
    data['ExtDesc'] = data['Descriptors'].apply(lambda w: get_extDesc(w, analyzeTool))

    # Add Iterative Split
    with open(datasetSplit + '/train.txt') as fin:
        trainset = [line.strip() for line in fin]
    with open(datasetSplit + '/validation.txt') as fin:
        valset = [line.strip() for line in fin]
    with open(datasetSplit + '/test.txt') as fin:
        testset = [line.strip() for line in fin]
    data['split'] = data['celex_id'].apply(lambda w: getSplit(w, trainset, valset, testset))

    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download and prepare JRC-Aquis dataset")
    parser.add_argument("--languages",
                        default="en",
                        help="supported languages separated by commas: \n"
                             + "List of supported Languages: en, de, it, fr", type=str)
    parser.add_argument("--dataset_split", help="path to dataset split files", type=str)
    parser.add_argument("--save_path", help="path of the dataset", type=str)

    args = parser.parse_args()

    languages = args.languages
    datasetSplit = args.dataset_split
    save_path = args.save_path

    prepareDataset(languages, datasetSplit, save_path)
