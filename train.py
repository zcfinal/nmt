from __future__ import print_function
import os
import sys
import time
import math
import argparse

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from data import data_utils
from data.dataset import get_dataloader
from transformer.models import Transformer
from transformer.optimizer import ScheduledOptimizer
from utils import parse_args, setuplogging
import logging

use_cuda = torch.cuda.is_available()


def create_model(args):
    args.src_vocab_size = args.src_vocab_size
    args.tgt_vocab_size = args.tgt_vocab_size

    logging.info('Creating new model parameters..')
    model = Transformer(args)  # Initialize a model state.
    model_state = {'args': args, 'curr_epochs': 0, 'train_steps': 0}

    if os.path.exists(args.model_path):
        logging.info('Reloading model parameters..')
        model_state = torch.load(args.model_path)
        model.load_state_dict(model_state['model_params'])

    if use_cuda:
        logging.info('Using GPU..')
        model = model.cuda()

    return model, model_state


def main(args):
    logging.info('Loading training and development data..')
    train_iter, dev_iter = get_dataloader(args)
    model, model_state = create_model(args)
    init_epoch = model_state['curr_epochs']
    if init_epoch >= args.max_epochs:
        logging.info('Training is already complete.',
              'current_epoch:{}, max_epoch:{}'.format(init_epoch, args.max_epochs))
        sys.exit(0)

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
    optimizer = ScheduledOptimizer(optim.Adam(model.trainable_params(), betas=(0.9, 0.98), eps=1e-9),
                                   args.d_model, args.n_layers, args.n_warmup_steps)
    best_eval = 1e8
    for epoch in range(init_epoch + 1, args.max_epochs + 1):
        train_loss, train_sents = train(model, criterion, optimizer, train_iter, model_state)
        logging.info('END Epoch {}\n'.format(epoch)+ 'Train_ppl: {0:.2f}\n'.format(train_loss)+ \
              'Sents seen: {}\n'.format(train_sents))
        
        eval_loss, eval_sents = eval(model, criterion, dev_iter)
        logging.info('END Epoch {}\n'.format(epoch)+ 'Eval_ppl: {0:.2f}\n'.format(eval_loss)+ \
              'Sents seen: {}\n'.format(eval_sents))

        model_state['curr_epochs'] += 1
        model_state['model_params'] = model.state_dict()
        torch.save(model_state, args.model_path)
        if eval_loss<best_eval:
            best_eval = eval_loss
            torch.save(model_state,args.log+'/best.pth')
        logging.info('The model checkpoint file has been saved')


def train(model, criterion, optimizer, train_iter, model_state):  # TODO: fix opt
    model.train()
    args = model_state['args']
    train_loss, train_loss_total = 0.0, 0.0
    n_words, n_words_total = 0, 0
    n_sents, n_sents_total = 0, 0
    start_time = time.time()
    for src,tgt in train_iter:
        enc_inputs = src.cuda()
        enc_inputs_len = src.shape[1]
        dec_ = tgt.cuda()
        dec_inputs_len = tgt.shape[1]
        dec_inputs = dec_[:, :-1]
        dec_targets = dec_[:, 1:]
        dec_inputs_len = dec_inputs_len - 1

        optimizer.zero_grad()
        dec_logits, _, _, _ = model(enc_inputs, enc_inputs_len,
                                    dec_inputs, dec_inputs_len)
        step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))

        step_loss.backward()
        if args.max_grad_norm:
            clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        optimizer.update_lr()

        train_loss_total += float(step_loss.item())
        n_words_total += torch.sum((~ dec_inputs.data.eq(0))).item()
        n_sents_total += dec_inputs.shape[0]
        model_state['train_steps'] += 1

        if model_state['train_steps'] % args.display_freq == 0:
            loss_int = (train_loss_total - train_loss)
            n_words_int = (n_words_total - n_words)
            n_sents_int = (n_sents_total - n_sents)

            loss_per_words = loss_int / n_words_int
            avg_ppl = math.exp(loss_per_words) if loss_per_words < 300 else float("inf")
            time_elapsed = (time.time() - start_time)
            step_time = time_elapsed / args.display_freq

            n_words_sec = n_words_int / time_elapsed
            n_sents_sec = n_sents_int / time_elapsed

            logging.info('Epoch {0:<3}\n'.format(model_state['curr_epochs'])+'Step {0:<10}\n'.format(model_state['train_steps'])+ \
                  'Perplexity {0:<10.2f}\n'.format(avg_ppl)+ 'Step-time {0:<10.2f}\n'.format(step_time)+ \
                  '{0:.2f} sents/s\n'.format(n_sents_sec)+ '{0:>10.2f} words/s\n'.format(n_words_sec) + f'{(train_loss_total / n_words_total)} train loss\n')
            train_loss, n_words, n_sents = (train_loss_total, n_words_total, n_sents_total)
            start_time = time.time()

    return math.exp(train_loss_total / n_words_total), n_sents_total


def eval(model, criterion, dev_iter):
    model.eval()
    eval_loss_total = 0.0
    n_words_total, n_sents_total = 0, 0

    logging.info('Evaluation')
    with torch.no_grad():
        for src,tgt in dev_iter:
            enc_inputs = src.cuda()
            enc_inputs_len = src.shape[1]
            dec_ = tgt.cuda()
            dec_inputs_len = tgt.shape[1]
            dec_inputs = dec_[:, :-1]
            dec_targets = dec_[:, 1:]
            dec_inputs_len = dec_inputs_len - 1

            dec_logits, *_ = model(enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len)
            step_loss = criterion(dec_logits, dec_targets.contiguous().view(-1))
            eval_loss_total += float(step_loss.item())
            n_words_total += torch.sum((~ dec_inputs.data.eq(0))).item()
            n_sents_total += dec_inputs.shape[0]
            logging.info('  {} samples seen'.format(n_sents_total))

    return math.exp(eval_loss_total / n_words_total), n_sents_total


if __name__ == '__main__':
    args = parse_args()
    setuplogging(args)
    logging.info(args)
    main(args)
    logging.info('Terminated')
