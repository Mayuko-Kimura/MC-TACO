#!/usr/bin/env bash

python experiments/bert/run_lm_finetuning_att_both.py \
       --train_file data_mlm/mlm_input_yes2.txt \
       --bert_model bert-base-uncased \
       --do_lower_case \
       --do_train \
       --output_dir mlm \
       --tfidf att_swag-mctaco_30.txt
