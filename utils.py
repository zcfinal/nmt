import argparse
import logging
import os
import sys
import torch
import random
import numpy as np

def setuplogging(args, rank=0):
    root = logging.getLogger()
    if len(root.handlers)<=1:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(f"[{rank}] [%(levelname)s %(asctime)s] %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)

        fh = logging.FileHandler(os.path.join(args.log,'logging_file.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Training Hyperparams')

    # network params
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-share_proj_weight', action='store_true')
    parser.add_argument('-share_embs_weight', action='store_true')
    parser.add_argument('-weighted_model', action='store_true')

    # training params
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-max_epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=50)
    parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    parser.add_argument('-log', default=None)
    parser.add_argument('-model_path', type=str, required=True)

    #preprocess
    parser.add_argument('--train_src', required=True, type=str, help='Path to training source data')
    parser.add_argument('--train_tgt', required=True, type=str, help='Path to training target data')
    parser.add_argument('--test_src', required=True, type=str, help='Path to test source data')
    parser.add_argument('--test_tgt', required=True, type=str, help='Path to test target data')
    parser.add_argument('--src_name', type=str )
    parser.add_argument('--tgt_name', type=str )
    parser.add_argument('--src_vocab_size', type=int, help='Source vocabulary size')
    parser.add_argument('--tgt_vocab_size', type=int, help='Target vocabulary size')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--lower_case', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    return args