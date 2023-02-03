#!/usr/bin/env bash

python experiments/roberta/run_swag.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data_swag \
  --bert_model roberta-large \
  --max_seq_length 256 \
  --train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 2.0 \
  --output_dir ./roberta_output \
  --output_model_file roberta_swag.pt \
