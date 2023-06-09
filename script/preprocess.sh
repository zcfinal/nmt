cd ../data

python dataset.py \
--train_src '/data/zclfe/transformer/corpora/training.en' \
--train_tgt '/data/zclfe/transformer/corpora/training.zh' \
--test_src '/data/zclfe/transformer/corpora/test_news_filtered.en' \
--test_tgt '/data/zclfe/transformer/corpora/test_news_filtered.zh' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 50 \
--src_name eng \
--tgt_name zh 