cd ..

CUDA_VISIBLE_DEVICES=6 python train.py \
--train_src '/data/zclfe/transformer/corpora/training.en' \
--train_tgt '/data/zclfe/transformer/corpora/training.zh' \
--test_src '/data/zclfe/transformer/corpora/test.en' \
--test_tgt '/data/zclfe/transformer/corpora/test.zh' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 50 \
--src_name eng \
--tgt_name zh \
-model_path '/data/zclfe/transformer/output/last.pth' \
-log '/data/zclfe/transformer/output' \
-display_freq 500 \
-lr 5e-5 \
-max_epochs 100 \
-n_layers 8