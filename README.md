# Legal-Docs-Large-MLTC

## 1. Prepare datasets

The following script will:
- Download JRC-Aquis data.
- Download and prepare EuroVoc data and EuroVoc Analysis tool.
- Prepare JRC-Aquis datasets for monolingual and multi-lingual experiments.

```
bash preparedata.sh
```

**OUTPUT FILES**

- JRC Monolingual Dataset: datasets/jrc_en_basic.csv
- JRC Multilingul Dataset: datasets/jrc_3langs_basic.csv
- EURLEX57K Dataset: datasets/EurLex57K.csv
- Pickle of EuroVocAnalyzeTool instance: data/EuroVocAnalysisTool.pickle
  - This instance store EuroVoc data.
  - This object provides access to relations within EuroVoc hierarchy.

## 2. Iterative split

Default splits for the datasets are provided in Iterative_Split/JRC_Aquis and Iterative_Split/EurLex57K
To make another split using iterative split approach, follow instructions in Iterative_Split/README.md

## TODO

- [ ] perepare and preprocess dataset
  - [X] JRC-Aquis dataset
  - [ ] EurLex57K 
- [X] Hierachical Reduction / extract MT/TT/DO information
- [ ] LM Finetuning & Classifier Finetuning
  - [ ] AWD-LSTM
  - [ ] Transformers 
  - [ ] bash scripts
- [ ] logs&results
