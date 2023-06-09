cd ..

CUDA_VISIBLE_DEVICES=2 python translate.py \
--train_src '/data/zclfe/transformer/corpora/training.en' \
--train_tgt '/data/zclfe/transformer/corpora/training.zh' \
--test_src '/data/zclfe/transformer/corpora/test_news_filtered.en' \
--test_tgt '/data/zclfe/transformer/corpora/test_news_filtered.zh' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 100 \
--src_name eng \
--tgt_name zh \
-model_path '/data/zclfe/transformer/output/best.pth' \
-decode_output '/data/zclfe/transformer/output/out_news.txt' \
-display_freq 500 \
-lr 5e-5 \
-max_epochs 100 \
-n_layers 6 \
-batch_size 1