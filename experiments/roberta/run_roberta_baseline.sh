#!/usr/bin/env bash

python experiments/roberta/run_classifier.py \
  --task_name TEMPORAL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir dataset \
  --bert_model roberta-large \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 10.0 \
  --output_dir ./roberta_output \
  --load_checkpoint mlm_roberta_v3.pt \
#  --output_dir ./roberta_output  
