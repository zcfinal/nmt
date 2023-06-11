cd ..

CUDA_VISIBLE_DEVICES=6 python train.py \
--train_src '/data/zclfe/transformer/corpora/train_en.txt' \
--train_tgt '/data/zclfe/transformer/corpora/train_zh.txt' \
--test_src '/data/zclfe/transformer/corpora/test_en.txt' \
--test_tgt '/data/zclfe/transformer/corpora/test_zh.txt' \
--src_vocab_size 5000 \
--tgt_vocab_size 5000 \
--max_len 64 \
--src_name eng_small \
--tgt_name zh_small \
-model_path '/data/zclfe/transformer/output/ls_6/last.pth' \
-log '/data/zclfe/transformer/output/ls_6' \
-display_freq 100 \
-lr 5e-5 \
-max_epochs 100 \
-n_layers 6 \
-decode_output '/data/zclfe/transformer/output/ls_6/out.txt' \
-dropout 0.1 \
-batch_size 128 \
-share_proj_weight