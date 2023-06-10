import torch
from torch.utils.data import Dataset, DataLoader, Subset
import sentencepiece as spm
import os
import argparse
import random

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
    parser.add_argument('-model_path', type=str)

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

    args = parser.parse_args()

    return args

class Tokenizer:
    def __init__(self,corpora_path,vocab_size=32000,model_name='eng',model_type='bpe',coverage=1,token_path=None) -> None:
        self.corpora_path = corpora_path
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.model_type = model_type
        self.coverage = coverage
        sp = spm.SentencePieceProcessor()
        if not os.path.exists(token_path):
            self.train(self.corpora_path, self.vocab_size, self.model_name, self.model_type, self.coverage)
        sp.Load(token_path)
        self.vocab = sp
    
    def encode(self,text):
        return self.vocab.EncodeAsIds(text,add_bos=True,add_eos=True,emit_unk_piece=True)

    def decode(self,input):
        return self.vocab.Decode(input)

    def train(self, input_file, vocab_size, model_name, model_type, character_coverage):
        input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                    '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
        cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
        spm.SentencePieceTrainer.Train(cmd)

class CorporaSet(Dataset):
    def __init__(self, args, data_path, vocab_size=32000, model_name='eng', model_type='bpe', coverage=1):
        super().__init__()
        self.args = args
        self.tokenize = Tokenizer(data_path,vocab_size,model_name,model_type,coverage,f'./data/{model_name}.model')
        self.data_th = data_path+'.pth'
        if not os.path.exists(self.data_th):
            self.preprocess(data_path)
            self.need_load = True
        else:
            self.tok_data = torch.load(self.data_th)
            self.need_load = False
    
    def preprocess(self,origin_path):
        self.tok_data = []
        with open(origin_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tid = self.tokenize.encode(line)
                self.tok_data.append(tid)
    def __getitem__(self, index):
        return self.tok_data[index]

    def __len__(self):
        return self.tok_data.shape[0]




class TranslationDataset(Dataset):
    def __init__(self, source_datapath, target_datapath, args):
        super().__init__()
        self.args = args
        self.source_corpora = CorporaSet(args,source_datapath,args.src_vocab_size,args.src_name)
        self.target_corpora = CorporaSet(args,target_datapath,args.tgt_vocab_size,args.tgt_name)
        if self.source_corpora.need_load:
            self.filter_save()
    
    def filter_save(self):
        filtered_src_data = []
        filtered_tgt_data = []
        for src_line, tgt_line in zip(self.source_corpora.tok_data,self.target_corpora.tok_data):
            if len(src_line)<=self.args.max_len and len(tgt_line)<=self.args.max_len:
                src_line = src_line+[0]*(self.args.max_len - len(src_line))
                filtered_src_data.append(torch.LongTensor(src_line))
                tgt_line = tgt_line+[0]*(self.args.max_len - len(tgt_line))
                filtered_tgt_data.append(torch.LongTensor(tgt_line))
        self.source_corpora.tok_data = torch.stack(filtered_src_data,0)
        self.target_corpora.tok_data = torch.stack(filtered_tgt_data,0)

        torch.save(self.source_corpora.tok_data,self.source_corpora.data_th)
        torch.save(self.target_corpora.tok_data,self.target_corpora.data_th)
    
    def __getitem__(self, index):
        src = self.source_corpora.__getitem__(index)
        tgt = self.target_corpora.__getitem__(index)
        
        return (src,tgt)

    def __len__(self):
        return len(self.source_corpora)

def collate_fn(batch):
    source_sentences, target_sentences = zip(*batch)
    source_sentences = torch.stack(source_sentences,0)
    target_sentences = torch.stack(target_sentences,0)
    return source_sentences, target_sentences

def get_dataloader(args):
    dataset = TranslationDataset(args.train_src,args.train_tgt,args)
    data_idx = list(range(len(dataset)))
    random.shuffle(data_idx)
    train_datalen = int(len(dataset)*0.9)
    train_idx = data_idx[:train_datalen]
    dev_idx = data_idx[train_datalen:]
    train_dataset = Subset(dataset, train_idx)
    dev_dataset = Subset(dataset, dev_idx)    
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=True)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=False)
    return train_dataloader,dev_dataloader

def get_testdataloader(args):
    dataset = TranslationDataset(args.test_src,args.test_tgt,args)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=False)
    return dataloader, dataset.target_corpora.tokenize


if __name__=='__main__':
    args = parse_args()
    TranslationDataset(args.train_src,args.train_tgt,args)
    #get_testdataloader(args)