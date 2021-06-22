#! /bin/bash

# 1. preprocess dataset by the following. It will produce data/sparc_data_removefrom/
devices=$1
GLOVE_PATH="../data/glove.840B.300d.txt" # you need to change this
LOGDIR=$2

python3 preprocess.py --dataset=sparc --remove_from > ${LOGDIR}.data.log 2>&1 &
wait
# 2. train and evaluate.
#    the result (models, logs, prediction outputs) are saved in $LOGDIR
if [ ! -d ${LOGDIR} ]; then
  mkdir ${LOGDIR}
fi
wait
cp run_sparc_editsql.sh ./${LOGDIR}/
wait
cp ./model/schema_interaction_model.py ./${LOGDIR}/
wait
CUDA_VISIBLE_DEVICES=$devices python3 run.py --raw_train_filename="../data/sparc_data_removefrom/train.pkl" \
          --raw_validation_filename="../data/sparc_data_removefrom/dev.pkl" \
          --database_schema_filename="../data/sparc_data_removefrom/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_sparc_removefrom" \
          --input_key="utterance" \
          --use_copy_switch=1 \
          --use_schema_encoder=1 \
          --use_bert=1 \
          --bert_type_abb=uS \
          --fine_tune_bert=1 \
          --interaction_level=1 \
          --use_predicted_queries=1 \
          --copy_switch_query=1 \
          --freeze=1 \
          --train=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
\
          --use_previous_query=1 \
          --use_query_attention=1 \
          --use_schema_attention=1 \
          --dst_loss_weight=0.0 \
\
          --reweight_batch=1 \
          --get_seq_from_scores_flatten=1 \
\
          --norm_loss=1 \
          --evaluate_split="valid" > ${LOGDIR}.model.log 2>&1 &
wait
# 3. get evaluation result
python3 postprocess_eval.py --dataset=sparc --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from > ${LOGDIR}.eval.log 2>&1 &
