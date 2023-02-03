#!/usr/bin/env bash

python experiments/albert/run_classifier.py \
  --task_name TEMPORAL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir dataset \
  --bert_model albert-xxlarge-v2 \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5.0 \
  --load_checkpoint albert_swag_256_16_1e-5_epo1_trimmed.pt \
  --output_dir ./albert_output 
