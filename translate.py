# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from utils import parse_args

from data.dataset import get_testdataloader
from transformer.translator import Translator

use_cuda = torch.cuda.is_available()


def main(args):
    translator = Translator(args, use_cuda)

    
    test_dataloader, tokenize = get_testdataloader(args)

    lines = 0
    print('Translated output will be written in {}'.format(args.decode_output))
    with open(args.decode_output, 'w') as output:
        with torch.no_grad():
            for src,tgt in test_dataloader:
                all_hyp, all_scores = translator.translate_batch(src)
                for idx_seqs in all_hyp:
                    for idx_seq in idx_seqs:
                        pred_line = tokenize.decode(idx_seq)
                        origin_line = tokenize.decode(tgt.tolist()[0])
                        output.write('machine:\n'+pred_line + '\norigin:\n'+origin_line+'\n')
                        output.flush()
                lines += src.shape[0]
                if lines % 1000==0:
                    print('  {} lines decoded'.format(lines))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    print('Terminated')