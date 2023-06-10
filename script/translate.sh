cd ..

CUDA_VISIBLE_DEVICES=3 python translate.py \
--train_src '/data/zclfe/transformer/corpora/datum_en_train.txt' \
--train_tgt '/data/zclfe/transformer/corpora/datum_zh_train.txt' \
--test_src '/data/zclfe/transformer/corpora/datum_en_test.txt' \
--test_tgt '/data/zclfe/transformer/corpora/datum_zh_test.txt' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 64 \
--src_name eng_large \
--tgt_name zh_large \
-model_path '/data/zclfe/transformer/output/large/best.pth' \
-decode_output '/data/zclfe/transformer/output/large/out.txt' \
-display_freq 500 \
-lr 5e-5 \
-max_epochs 100 \
-n_layers 6 \
-batch_size 1