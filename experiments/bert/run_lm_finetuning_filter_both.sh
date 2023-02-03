#!/usr/bin/env bash

python experiments/bert/run_lm_finetuning_filter_both.py \
       --train_file data_mlm/mlm_input_yes.txt \
       --bert_model bert-base-uncased \
       --do_lower_case \
       --do_train \
       --output_dir mlm \
       --tfidf tfidf_0.25_nosklearn_stopwords_time.txt
