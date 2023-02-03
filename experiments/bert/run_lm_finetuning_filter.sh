#!/usr/bin/env bash

python experiments/bert/run_lm_finetuning_filter.py \
  --train_file data_mlm/mlm_input_yes.txt \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --do_train \
  --output_dir mlm \
  --tfidf tfidf_top_half_nosklearn_stopwords.txt  \
  #--pos_filter 
