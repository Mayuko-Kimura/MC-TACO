#!/usr/bin/env bash

python experiments/roberta/run_lm_finetuning.py \
  --train_file data_mlm/mlm_input_yes.txt \
  --bert_model roberta-base \
  --do_lower_case \
  --do_train \
  --output_dir mlm_roberta 
