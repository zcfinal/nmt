cd ../data

python dataset.py \
--train_src '/data/zclfe/transformer/corpora/datum_en_train.txt' \
--train_tgt '/data/zclfe/transformer/corpora/datum_zh_train.txt' \
--test_src '/data/zclfe/transformer/corpora/datum_en_test.txt' \
--test_tgt '/data/zclfe/transformer/corpora/datum_zh_test.txt' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 64 \
--src_name eng_large \
--tgt_name zh_large 
