DO_TRAIN=1
cased=0

dataset_path='../../datasets/EurLex57K.csv'
dataset_split_path='../../Iterative_Split/EurLex57K/'
pretrained_model_name="/home/zein/EurLex/LM-finetune/RoBERTa-LM-finetuned-EurLex"
model_type="roberta"

TOTAL_CYCLES=5
N_ITERATIONS="4,4,4,12,12"
MAX_LR="2e-04,5e-05,5e-05,5e-05,5e-05"
MAX_LEN=512
UNFREEZED="-4,-8,-12"
LABEL_COL_NAME="Labels" # Descriptors ExtDesc Domains MThesaurus
experiment_name="EurLex57K-Iterative-roberta" # "01" , "04-Topterm"

START_CYCLE=1
lr_find=0

if [ $DO_TRAIN -eq 1 ];
then
    echo "Training...";
    python finetune.py --LABEL_COL_NAME=$LABEL_COL_NAME \
                        --experiment_name=$experiment_name \
                        --dataset_path=$dataset_path \
                        --dataset_split_path=$dataset_split_path \
                        --pretrained_model_name=$pretrained_model_name \
                        --model_type=$model_type \
                        --cased=$cased \
                        --LR=$MAX_LR --MAX_LEN=$MAX_LEN \
                        --N_ITERATIONS=$N_ITERATIONS \
                        --UNFREEZED=$UNFREEZED --TOTAL_CYCLES=$TOTAL_CYCLES \
                        --START_CYCLE=$START_CYCLE --lr_find=$lr_find
fi

# python inference.py --LABEL_COL_NAME=$LABEL_COL_NAME \
#                     --model_output_name=$model_output_name \
#                     --dataset_path=$dataset_path \
#                     --pretrained_model_name=$pretrained_model_name \
#                     --model_type=$model_type \
#                     --MAX_LEN=$MAX_LEN
                    
# python evaluate_results.py --LABEL_COL_NAME=$LABEL_COL_NAME \
#                            --experiment=$model_output_name \
#                            --title=$LABEL_COL_NAME 
#                            # \--use_dynamic_thresholding
