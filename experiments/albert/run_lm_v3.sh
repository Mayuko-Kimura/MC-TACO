#!/usr/bin/env bash

python experiments/albert/run_lm_v3.py \
  --train_data_file data_mlm/mlm_input_yes2.txt \
  --model_name_or_path albert-xxlarge-v2 \
  --do_train \
  --output_dir mlm_albert \
  --line_by_line \
  --mlm

