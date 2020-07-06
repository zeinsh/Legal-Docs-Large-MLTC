# Language Model Finetuning


## Writing finetuning script

1) Prepare finetuning parameters.

```
batch_size=4 # Batch size for transformer-based model
num_train_epochs=5 # number of iterations / epochs
save_total_limit=5 # keep last 5 checkpoints
save_steps=3000  # save checkpoint every 3000 global steps

dataset_name="jrc_en" 
dataset_path="../../datasets/jrc_en_basic.csv" # path to dataset

model_type="distilbert" # model_type: bert, roberta, distilbert
MODEL_PATH="distilbert-base-uncased" # pretrained model name

OUTPUT_DIR="./lm-finetuned/"$dataset_name"/"$model_type # where to save finetuned model
TRAIN_FILE=tmp/train.txt # temporary text file to save training texts
TEST_FILE=tmp/test.txt # temporary text file to save test texts

```

2) Prepare text data from the dataset to finetune LM.


```
python prepareTextData.py --dataset_path=$dataset_path \
                          --train_output_path=$TRAIN_FILE \
                          --test_output_path=$TEST_FILE
```

3) Run LM Finetuning.

```
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
```

## Example:

Run LM finetuning for DistilBert. **Note:** Run the script from within this directory.

```
$ bash ./runs/distilbert-jrc_en-20200624.sh 
```