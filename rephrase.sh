# 扩充文本匹配的语料  文本复述任务
#　成都
# pyenv activate python373tf115
pip install install tensorflow-gpu==1.15.4 bert-tensorflow==1.0.1
#python -m pip install --upgrade pip -i https://pypi.douban.com/simple
export PATH="/home/aistudio/lib/;"+$PATH
# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# 房山
# pyenv activate python363tf111
# pip install bert-tensorflow==1.0.1

#scp -r /home/cloudminds/PycharmProjects/lasertagger-Chinese/predict_main.py  cloudminds@10.13.33.128:/home/cloudminds/PycharmProjects/lasertagger-Chinese
#scp -r cloudminds@10.13.33.128:/home/wzk/Mywork/corpus/文本复述/output/models/wikisplit_experiment_name /home/cloudminds/Mywork/corpus/文本复述/output/models/
#watch -n 1 nvidia-smi

start_tm=`date +%s%N`;

export HOST_NAME="wzk" #"cloudminds" #　 　
### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=wikisplit_experiment
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
NUM_EPOCHS=1000000
export TRAIN_BATCH_SIZE=1  # 512 OOM   256 OK
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=200
export enable_swap_tag=false
export output_arbitrary_targets_for_infeasible_examples=false
export WIKISPLIT_DIR="/content/text_data_enhancement_with_LaserTagger/data/"
export OUTPUT_DIR="/content/drive/Shareddrives/paperdrive/rephrase/output"
cd /content/text_data_enhancement_with_LaserTagger
python phrase_vocabulary_optimization.py \
 --input_file=${WIKISPLIT_DIR}/train.txt \
 --input_format=wikisplit \
 --vocabulary_size=500 \
 --max_input_examples=1000000 \
 --enable_swap_tag=${enable_swap_tag} \
 --output_file=${OUTPUT_DIR}/label_map.txt


export max_seq_length=512 # TODO
export BERT_BASE_DIR="/content/RoBERTa" # chinese_L-12_H-768_A-12"

python preprocess_main.py \
 --input_file=${WIKISPLIT_DIR}/tune.txt \
 --input_format=wikisplit \
 --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
 --label_map_file=${OUTPUT_DIR}/label_map.txt \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --max_seq_length=${max_seq_length} \
 --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}  # TODO true

python preprocess_main.py \
   --input_file=${WIKISPLIT_DIR}/train.txt \
   --input_format=wikisplit \
   --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
   --label_map_file=${OUTPUT_DIR}/label_map.txt \
   --vocab_file=${BERT_BASE_DIR}/vocab.txt \
   --max_seq_length=${max_seq_length} \
   --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples} # TODO false



# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=41624
export NUM_EVAL_EXAMPLES=5000
export CONFIG_FILE=configs/lasertagger_config.json
export EXPERIMENT=wikisplit_experiment_name



# python run_lasertagger.py \
#  --training_file=${OUTPUT_DIR}/train.tf_record \
#  --eval_file=${OUTPUT_DIR}/tune.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
#  --do_train=true \
#  --do_eval=true \
#  --train_batch_size=${TRAIN_BATCH_SIZE} \
#  --save_checkpoints_steps=200 \
#  --max_seq_length=${max_seq_length} \
#  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
#  --num_eval_examples=${NUM_EVAL_EXAMPLES}

CUDA_VISIBLE_DEVICES=0 nohup python run_lasertagger.py \
 --training_file=${OUTPUT_DIR}/train.tf_record \
 --eval_file=${OUTPUT_DIR}/tune.tf_record \
 --label_map_file=${OUTPUT_DIR}/label_map.txt \
 --model_config_file=${CONFIG_FILE} \
 --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
 --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
 --do_train=true \
 --do_eval=true \
 --train_batch_size=${TRAIN_BATCH_SIZE} \
 --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
 --num_train_epochs=${NUM_EPOCHS} \
 --max_seq_length=${max_seq_length} \
 --num_train_examples=${NUM_TRAIN_EXAMPLES} \
 --num_eval_examples=${NUM_EVAL_EXAMPLES} > log.txt 2>&1 &


### 4. Prediction

# Export the model.
python run_lasertagger.py \
 --label_map_file=${OUTPUT_DIR}/label_map.txt \
 --model_config_file=${CONFIG_FILE} \
 --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
 --do_export=true \
 --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export

### Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv

python predict_main.py \
 --input_file=${WIKISPLIT_DIR}/test.txt \
 --input_format=wikisplit \
 --output_file=${PREDICTION_FILE} \
 --label_map_file=${OUTPUT_DIR}/label_map.txt \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --max_seq_length=${max_seq_length} \
 --saved_model=${SAVED_MODEL_DIR}

#### 5. Evaluation
python score_main.py --prediction_file=${PREDICTION_FILE}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"