export CUDA_VISIBLE_DEVICES=0

project_dir="/home/zein/Legal-Docs-Large-MLTC"

batch_size=4
num_train_epochs=5
save_total_limit=5
save_steps=3000
model_type="bert"
dataset_name="jrc_3langs"
dataset_path=$project_dir"/datasets/jrc_3langs_basic.csv"
MODEL_PATH="/home/zein/Legal-Docs-Large-MLTC/training/LM-finetuning/lm-finetuned/eurlex_mul/bert5EPOCH_LMFT"

OUTPUT_DIR="./lm-finetuned/"$dataset_name"/"$model_type"Eurlex5X+Jrc5X-5EPOCH_LMFT"
TRAIN_FILE=tmp/train.txt
TEST_FILE=tmp/test.txt


python prepareTextData.py --dataset_path=$dataset_path \
                          --train_output_path=$TRAIN_FILE \
                          --test_output_path=$TEST_FILE

python run_lm_finetuning.py \
--model_name_or_path=$MODEL_PATH \
--train_file=$TRAIN_FILE \
--validation_file=$TEST_FILE \
--do_train \
--do_eval \
--output_dir=$OUTPUT_DIR \
--per_device_train_batch_size $batch_size \
--save_total_limit $save_total_limit \
--save_steps $save_steps \
--num_train_epochs $num_train_epochs \
--overwrite_output_dir
