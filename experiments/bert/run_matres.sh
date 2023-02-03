#!/usr/bin/env bash

python experiments/bert/run_classifier.py \
       --task_name MATRES \
       --do_train \
       --do_eval \
       --do_lower_case \
       --data_dir data \
       --bert_model bert-base-uncased \
       --max_seq_length 128 \
       --train_batch_size 16 \
       --learning_rate 1e-5 \
       --num_train_epochs 3.0 \
       --save_model matres.pt \
       --output_dir ./bert_output_matres
