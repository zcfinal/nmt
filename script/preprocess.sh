cd ../data

python dataset.py \
--train_src '/data/zclfe/transformer/corpora/train_en.txt' \
--train_tgt '/data/zclfe/transformer/corpora/train_zh.txt' \
--test_src '/data/zclfe/transformer/corpora/test_en.txt' \
--test_tgt '/data/zclfe/transformer/corpora/test_zh.txt' \
--src_vocab_size 5000 \
--tgt_vocab_size 5000 \
--max_len 64 \
--src_name eng_small \
--tgt_name zh_small
