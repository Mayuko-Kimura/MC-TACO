# MC-TACO

MC-TACOで評価をする時に使用したコードです。
一段階目に補助データセットを使用した多段階ファインチューニング、対象タスクに対する継続学習として、MC-TACOを用いたMLM、
またBERTを用いてMLMにおけるマスクの方法を工夫するコードなどがあります。

## dataset

MC-TACOのデータセットを置いてあります。

・dev_3783.tsv：検証データ

・test_9442.tsv：評価データ

・dev_3783_1.tsv〜dev_3783_5.tsv：５分割交差検証用のデータ

## evaluator

MC-TACOでの評価スコアを計算するコード。

Exact Match（MC-TACO独自の評価指標）とF1スコアを計算して出力します。

計算方法は[MC-TACOの論文](https://aclanthology.org/D19-1332/)参照。

## experiments

BERT、RoBERTa、ALBERTの各モデルに対して、
一段階目に補助データセットを使用した多段階ファインチューニング、対象タスクに対する継続学習として、MC-TACOを用いたMLMを行うコード。
BERTについてはMLMにおけるマスクの方法を工夫するコードもあります。

### bert/
・run_classifier.py, run_bert_baseline.sh：MC-TACOでファインチューニング、評価

・run_bert_timebank.sh：TimeMLでファインチューニング（.pyファイルはrun_classifier.py）

・run_matres.sh：MATRESでファインチューニング（.pyファイルはrun_classifier.py）

・run_multiple_choice.py, run_bert_baseline_multiple_choice.sh：CosmosQAでファインチューニング

・run_swag.py, run_swag.sh：SWAGでファインチューニング

・trim_model.py, run_trim_model.sh：多段階ファインチューニングを行う際に、一段階目と二段階目でタスクの形式が異なる場合（CosmosQA, SWAGなど）のパラメータ調整用

### roberta/

### albert/



## 主なコマンド

ファインチューニングを行うとき：

`sh experiments/bert/run_bert_baseline.sh`

MC-TACOでの精度を計算するとき：

`python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file bert_output/eval_outputs.txt`

多段階ファインチューニングをするとき：

`sh experiments/bert/run_bert_baseline_multiple_choice.sh 

(sh experiments/bert/run_trim_model.sh)

sh experiments/bert/run_bert_baseline.sh`
