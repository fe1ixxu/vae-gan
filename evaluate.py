import time
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text

import os
import torch
import numpy as np
from torch import nn, optim


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text

def load_dataset(config, TEXT, test_file):
    
    dataset_fn = lambda name: data.TabularDataset(
        path=name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    test_pos_set, test_neg_set = map(dataset_fn, [test_file, test_file])

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
        
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)
    
    return test_iters

def auto_eval(config, vocab, model_F, test_iters, temperature):
    model_F.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    
    gold_text, raw_output, rev_output = inference(pos_iter, 1)
   
    # save output
    with open(config.output_file, 'w') as fw:
        for idx in range(len(rev_output)):
            print(rev_output[idx], file=fw)


def evaluator(config, TEXT, model_F, test_file):
    test_iters = load_dataset(config, TEXT, test_file)
    auto_eval(config, TEXT.vocab, model_F, test_iters, 1.0)