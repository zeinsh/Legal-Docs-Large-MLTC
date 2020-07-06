## Finetune classifier


The training script is like the following:

```
# dataset configurations
dataset_name="jrc_en"
dataset_path='../../datasets/jrc_en_basic.csv'
dataset_split_path='../../Iterative_Split/JRC_Aquis/'

# pretrained lm configurations
cased=1
model_type="distilbert"
pretrained_model_name="/home/zein/Legal-Docs-Large-MLTC/training/LM-finetuning/lm-finetuned/"$dataset_name"/"$model_type"_cased"

# language settings
trainLanguages="en" # you can set "en,de,fr" if available
testLanguages="en" # test only using English val/test sets

## Training configurations
START_CYCLE=1 # start/continue or restart from this cycle
TOTAL_CYCLES=9 
N_ITERATIONS="4,4,4,4,4,4,4,4,4" # number of iterations per cycle
MAX_LR="2e-04,5e-05,5e-05,5e-05,5e-05" # max learning rate per iteration
MAX_LEN=512
UNFROZEN="-2,-4,-6" # number of unfrozen layers per cycle (the rest of cycles will take the last value) 
LABEL_COL_NAME="Descriptors" # Descriptors ExtDesc Domains MThesaurus
experiment_name="baseline-512" 

lr_find=0 # 1 if you want to only fine learning rate

echo "Training...";
python finetune.py --LABEL_COL_NAME=$LABEL_COL_NAME \
                   --experiment_name=$experiment_name \
                   --dataset_name=$dataset_name \
                   --dataset_path=$dataset_path \
                   --dataset_split_path=$dataset_split_path \
                   --pretrained_model_name=$pretrained_model_name \
                   --model_type=$model_type \
                   --cased=$cased \
                   --LR=$MAX_LR --MAX_LEN=$MAX_LEN \
                   --N_ITERATIONS=$N_ITERATIONS \
                   --UNFREEZED=$UNFROZEN --TOTAL_CYCLES=$TOTAL_CYCLES \
                   --START_CYCLE=$START_CYCLE --lr_find=$lr_find \
                   --trainLanguages=$trainLanguages --testLanguages=$testLanguages 

```

The ourput will be saved in the following path: ```experiments/$dataset_name/$model_name/$experiment_name```