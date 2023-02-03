#!/usr/bin/env bash

python experiments/roberta/trim_model.py \
  --model_path roberta_cosmosqa_256_32_1e-5_epo3.pt \
  --target_model  roberta-large\
  --save_model roberta_cosmosqa_256_32_1e-5_epo3.pt_trimmed.pt \
