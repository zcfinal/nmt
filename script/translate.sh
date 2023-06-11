cd ..

CUDA_VISIBLE_DEVICES=3 python translate.py \
--train_src '/data/zclfe/transformer/corpora/train_en.txt' \
--train_tgt '/data/zclfe/transformer/corpora/train_zh.txt' \
--test_src '/data/zclfe/transformer/corpora/test_en.txt' \
--test_tgt '/data/zclfe/transformer/corpora/test_zh.txt' \
--src_vocab_size 5000 \
--tgt_vocab_size 5000 \
--max_len 64 \
--src_name eng_small \
--tgt_name zh_small \
-model_path '/data/zclfe/transformer/output/ls_6/best.pth' \
-decode_output '/data/zclfe/transformer/output/ls_6/out.txt' \
-display_freq 500 \
-lr 5e-5 \
-max_epochs 40 \
-n_layers 1 \
-batch_size 1 \
-beam_size 10