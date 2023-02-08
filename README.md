# MC-TACO

MC-TACOで評価をする時に使用したコードです。
一段階目に補助データセットを使用した多段階ファインチューニング、対象タスクに対する継続学習として、MC-TACOを用いたMLM、
またBERTを用いてMLMにおけるマスクの方法を工夫するコードなどがあります。

## dataset

MC-TACOのデータセットを置いてあります。

## evaluator

MC-TACOでの評価スコアを計算するコード。

Exact Match（MC-TACO独自の評価指標）とF1スコアを計算して出力します。

## experiments

BERT、RoBERTa、ALBERTの各モデルに対して、
一段階目に補助データセットを使用した多段階ファインチューニング、対象タスクに対する継続学習として、MC-TACOを用いたMLMを行うコード。
BERTについてはMLMにおけるマスクの方法を工夫するコードもあります。

## 主なコマンド

ファインチューニングを行うとき：

`sh experiments/bert/run_bert_baseline.sh`

MC-TACOでの精度を計算するとき：

`python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file bert_output/eval_outputs.txt`
