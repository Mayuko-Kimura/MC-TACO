#!/usr/bin/env bash

python experiments/albert/run_multiple_choice.py \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir data_cosmosqa \
  --bert_model albert-xxlarge-v2 \
  --max_seq_length 256 \
  --train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./albert_output_cosmosqa \
  --save_model albert_cosmosqa.pt
