#!/usr/bin/env bash

python experiments/bert/run_lm_finetuning.py \
  --train_file data_mlm/mlm_input_yes.txt \
  --bert_model bert-large-uncased \
  --do_lower_case \
  --do_train \
  --output_dir mlm 
