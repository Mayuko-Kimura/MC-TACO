#!/usr/bin/env bash

python experiments/bert/run_classifier.py \
  --task_name TIMEBANK \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --output_dir ./bert_output_timebank
