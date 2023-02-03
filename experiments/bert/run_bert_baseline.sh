#!/usr/bin/env bash

python experiments/bert/run_classifier.py \
  --task_name TEMPORAL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir dataset \
  --bert_model bert-large-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5.0 \
  --output_dir ./bert_output \
  --load_checkpoint mlm_bertlarge/mlm.pt
