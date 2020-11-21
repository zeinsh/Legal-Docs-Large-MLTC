export CUDA_VISIBLE_DEVICES=1

project_dir="/home/zein/Legal-Docs-Large-MLTC"

batch_size=8
num_train_epochs=5
save_total_limit=5
save_steps=3000
model_type="distilbert"
dataset_name="jrc_3langs"
dataset_path=$project_dir"/datasets/jrc_3langs_basic.csv"
MODEL_PATH="distilbert-base-multilingual-cased"

OUTPUT_DIR="./lm-finetuned/"$dataset_name"/"$model_type
TRAIN_FILE=tmp/train.txt
TEST_FILE=tmp/test.txt

python prepareTextData.py --dataset_path=$dataset_path \
                          --train_output_path=$TRAIN_FILE \
                          --test_output_path=$TEST_FILE

python run_lm_finetuning.py \
--per_device_train_batch_size $batch_size \
--save_total_limit $save_total_limit \
--save_steps $save_steps \
--num_train_epochs $num_train_epochs \
--output_dir=$OUTPUT_DIR \
--model_type=$model_type \
--model_name_or_path=$MODEL_PATH \
--do_train \
--train_data_file=$TRAIN_FILE \
--do_eval \
--eval_data_file=$TEST_FILE \
--mlm --overwrite_output_dir
