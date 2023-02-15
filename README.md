# MC-TACO

MC-TACOで評価をする時に使用したコードです。
一段階目に補助データセットを使用した多段階ファインチューニング、対象タスクに対する継続学習として、MC-TACOを用いたMLM、
またBERTを用いてMLMにおけるマスクの方法を工夫するコードなどがあります。

## dataset

MC-TACOのデータセットを置いてあります。

・dev_3783.tsv：検証データ

・test_9442.tsv：評価データ

・dev_3783_1.tsv〜dev_3783_5.tsv：５分割交差検証用のデータ

## data_mlm

MC-TACOを用いたMLM、BERTを用いてMLMにおけるマスクの方法を工夫するのに使用するデータセットの例を置いてあります。

・mlm_input_yes.txt：MLMに使用するために変形したMC-TACOの検証用データ。ラベルがnoのサンプルはノイズになって精度の低下につながるのでyesのもののみ使用。

・mlm_time.txt：時間関係の単語を優先してマスクする設定の時に使用するデータ。MC-TACOの各サンプルの後に、タブ区切りで時間関係の単語（優先してマスクしたい単語）を記載。

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

・run_lm_finetuning.py, run_lm_finetuning.sh：MC-TACOを用いたMLM（ランダムマスキング）

・run_lm_finetuning_filter.py, run_lm_finetuning_filter.sh：MC-TACOを用いたMLM（何かしらの基準を設定し、基準を満たす単語の多くをマスクする（満たさない単語はマスクしない））

・run_lm_finetuning_filter_both.py, run_lm_finetuning_filter_both.sh：MC-TACOを用いたMLM（何かしらの基準を設定し、基準を満たす単語の一部（多め）と基準を満たさない単語の一部（少なめ）をマスクする。基準を満たす単語を優先してマスクしたい場合）

### roberta/
基本はbert/と同じ

・run_lm_v3.py, run_lm_v3.sh：MC-TACOを用いたMLM（ランダムマスキング）


### albert/
bert/、roberta/と同じ


## 主なコマンド

ファインチューニングを行うとき：

`sh experiments/bert/run_bert_baseline.sh`

MC-TACOでの精度を計算するとき：

`python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file bert_output/eval_outputs.txt`

多段階ファインチューニングをするときの例：

`sh experiments/bert/run_bert_baseline_multiple_choice.sh`
（一段階目のファインチューニング）

(`sh experiments/bert/run_trim_model.sh`)

`sh experiments/bert/run_bert_baseline.sh`
（二段階目のMC-TACOでのファインチューニング）
