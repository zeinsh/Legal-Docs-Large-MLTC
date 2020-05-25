# Iterative Split

**iterativeSplit.py**

```
usage: Iterative split [-h] [--OUTPUT_DIR OUTPUT_DIR]
                       [--DATASET_PATH DATASET_PATH] [--LABEL LABEL]

optional arguments:
  -h, --help            show this help message and exit
  --OUTPUT_DIR OUTPUT_DIR
                        directory to save the splits
  --DATASET_PATH DATASET_PATH
                        path to the dataset
  --LABEL LABEL         Label type: Descriptors, Topterm, ExtDesc, MThesaurus,
                        Domains
```

## Example: 

```
python iterativeSplit.py --OUTPUT_DIR   ./newsplit/ \
                         --DATASET_PATH ../datasets/jrc_en_basic.csv \
                         --LABEL Descriptors
```