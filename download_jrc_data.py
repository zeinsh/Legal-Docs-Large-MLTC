import os
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from tqdm import tqdm


LANGUAGES = "en"
SAVE_PATH = "datasets/jrc_en_basic.csv"

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

def prepareDataset(languages, save_path):
    dir_path = SAVE_PATH[:save_path.rfind('/') + 1]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    COLNAMES=['filename', 'lang', 'year','text','Labels']
    data=pd.DataFrame(columns=COLNAMES)

    for lang in languages.split(","):
        for dirname in tqdm(os.listdir("./tmp/"+lang)):
            path="./tmp/"+lang+"/"+dirname
            if os.path.isdir(path):
                for filename in os.listdir(path):
                    celex_id=filename[3:-7]
                    try:
                        text, tagclasscode = parseXML(path, filename, sep=' #NP# ')
                        data.loc[len(data)]=[celex_id, lang, dirname, text, tagclasscode]
                    except Exception as ex:
                        print(ex)
                        print(path+'/'+filename)
    data.to_csv(save_path, index=False)

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser("Download and prepare JRC-Aquis dataset")
    parser.add_argument("--languages",
                        default="en",
                        help="supported languages separated by commas: \n"
                             + "List of supported Languages: en, de, it, fr", type=str)
    parser.add_argument("--save_path", help="path of the dataset", type=str)

    args = parser.parse_args()

    languages = args.languages
    save_path = args.save_path

    prepareDataset(languages, save_path)