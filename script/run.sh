cd ..

CUDA_VISIBLE_DEVICES=6 python train.py \
--train_src '/data/zclfe/transformer/corpora/datum_en_train.txt' \
--train_tgt '/data/zclfe/transformer/corpora/datum_zh_train.txt' \
--test_src '/data/zclfe/transformer/corpora/datum_en_test.txt' \
--test_tgt '/data/zclfe/transformer/corpora/datum_zh_test.txt' \
--src_vocab_size 32000 \
--tgt_vocab_size 32000 \
--max_len 64 \
--src_name eng_large \
--tgt_name zh_large \
-model_path '/data/zclfe/transformer/output/large/last.pth' \
-log '/data/zclfe/transformer/output/large' \
-display_freq 500 \
-lr 5e-5 \
-max_epochs 100 \
-n_layers 6 \
-decode_output '/data/zclfe/transformer/output/large/out.txt' 