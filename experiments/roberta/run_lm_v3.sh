#!/usr/bin/env bash

python experiments/roberta/run_lm_v3.py \
  --train_data_file data_mlm/mlm_input_yes2.txt \
  --model_name_or_path roberta-large \
  --do_train \
  --output_dir mlm_roberta \
  --line_by_line \
  --mlm
