#! /bin/bash

# 1. preprocess dataset by the following. It will produce data/cosql_data_removefrom/

# python3 preprocess.py --dataset=cosql --remove_from

# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR

GLOVE_PATH="../data/glove.840B.300d.txt" # you need to change this
LOGDIR="trained_model"

CUDA_VISIBLE_DEVICES=2 python3 run.py --raw_train_filename="../data/cosql_data_removefrom/train.pkl" \
          --raw_validation_filename="../data/cosql_data_removefrom/dev.pkl" \
          --database_schema_filename="../data/cosql_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_cosql_removefrom_test" \
          --input_key="utterance" \
          --use_copy_switch=1 \
          --use_schema_encoder=1 \
          --use_bert=1 \
          --bert_type_abb=uS \
          --fine_tune_bert=1 \
          --interaction_level=1 \
          --use_predicted_queries=1 \
          --freeze=1 \
          --logdir=$LOGDIR \
          --evaluate_split="valid" \
          --evaluate=1 \
\
          --use_previous_query=1 \
          --use_query_attention=1 \
          --use_schema_attention=1 \
          --get_seq_from_scores_flatten=1 \
          --dst_loss_weight=0.0 \
\
\
          --copy_switch_query=1 \
          --save_file="$LOGDIR/save_best"

# 3. get evaluation result

python3 postprocess_eval.py --dataset=cosql --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from
#          --use_previous_query=1 \
#          --use_query_attention=1 \
#          --use_schema_attention=1 \
#          --dst_loss_weight=1.0 \
#          --get_seq_from_scores_flatten=1 \
#          --new_loss=1 \
#          --use_previous_emb_for_previous_query=1 \
#          --use_current_emb_dynamic=1 \
