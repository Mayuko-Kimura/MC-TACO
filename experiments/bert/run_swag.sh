#!/usr/bin/env bash

python experiments/bert/run_swag.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data_swag \
  --bert_model bert-large-uncased \
  --max_seq_length 256 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./bert_output \
  --output_model_file swag.pt \
