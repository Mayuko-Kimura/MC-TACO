#!/usr/bin/env bash

python experiments/albert/trim_model.py \
  --model_path albert_swag_256_16_1e-5_epo1.pt \
  --target_model  albert-xxlarge-v2\
  --save_model albert_swag_256_16_1e-5_epo1_trimmed.pt \
