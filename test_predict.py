import bert_example
import predict_utils
import tagging_converter
import utils
import math, time
from termcolor import colored
import tensorflow as tf
'''
--input_file=${WIKISPLIT_DIR}/test.txt \
 --input_format=wikisplit \
 --output_file=${PREDICTION_FILE} \
 --label_map_file=${OUTPUT_DIR}/label_map.txt \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt \
 --max_seq_length=${max_seq_length} \
 --saved_model=${SAVED_MODEL_DIR}
'''
label_map_file = "${OUTPUT_DIR}/label_map.txt"
enable_swap_tag = True
vocab_file = "${BERT_BASE_DIR}/vocab.txt"
max_seq_length=40
do_lower_case=False
saved_model="${SAVED_MODEL_DIR}"
input_file=""
label_map = utils.read_label_map(label_map_file)
converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        enable_swap_tag)
builder = bert_example.BertExampleBuilder(label_map, vocab_file,
                                              max_seq_length,
                                              do_lower_case, converter)
predictor = predict_utils.LaserTaggerPredictor(
        tf.contrib.predictor.from_saved_model(saved_model), builder,
        label_map)
# print(colored("%s input file:%s" % (curLine(), FLAGS.input_file), "red"))
sources_list = []
target_list = []
# with tf.io.gfile.GFile(input_file) as f:
#     for line in f:
#         sources, target, lcs_rate = line.rstrip('\n').split('\t')
#         sources_list.append([sources])
#         target_list.append(target)
# number = len(sources_list)  # 总样本数
# predict_batch_size = min(64, number)
# batch_num = math.ceil(float(number) / predict_batch_size)
start_time = time.time()
num_predicted = 0

print(predictor.predict("Git 是一个开源的分布式版本控制系统，用于敏捷高效地处理任何或小或大的项目。"))
cost_time = (time.time() - start_time) / 60.0
print(cost_time)