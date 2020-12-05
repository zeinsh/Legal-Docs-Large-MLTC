export CUDA_VISIBLE_DEVICES=1

DO_TRAIN=1
cased=1

projectDir="/home/zein/Legal-Docs-Large-MLTC"
dataset_name="jrc_3langs"
dataset_path=$projectDir'/datasets/jrc_3langs_basic.csv'
dataset_split_path=$projectDir'/Iterative_Split/JRC_Aquis/'
pretrained_model_name=$projectDir"/training/LM-finetuning/lm-finetuned/jrc_3langs/distilbert1EPOCH_LMFT"
model_type="distilbert"

TOTAL_CYCLES=9
N_ITERATIONS="3,3,3,3,3,3,3,3,3"
MAX_LR="2e-04,5e-05,5e-05,5e-05,5e-05"
MAX_LEN=768
UNFREEZED="-2"
LABEL_COL_NAME="Descriptors" # Descriptors ExtDesc Domains MThesaurus
experiment_name="1LMFTtrainen-unfreeze2"

trainLanguages="en"
testLanguages="en,de,fr"

START_CYCLE=1
lr_find=0

if [ $DO_TRAIN -eq 1 ];
then
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
                        --UNFREEZED=$UNFREEZED --TOTAL_CYCLES=$TOTAL_CYCLES \
                        --START_CYCLE=$START_CYCLE --lr_find=$lr_find \
                        --trainLanguages=$trainLanguages --testLanguages=$testLanguages 
fi
