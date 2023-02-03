#!/usr/bin/env bash

python experiments/bert/trim_model.py \
  --model_path cosmosqa.pt \
  --target_model  bert-large-uncased\
  --save_model cosmosqa_256_32_2_1e-5_trimmed.pt \
