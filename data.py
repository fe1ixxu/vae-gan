import time
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text

class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text

def load_dataset(config, TEXT=None, train_pos='train.pos', train_neg='train.neg',
                 dev_pos='dev.pos', dev_neg='dev.neg',
                 test_pos='test.pos', test_neg='test.neg'):

    root = config.data_path
    text_flag = True if TEXT else False
    if not text_flag:
        TEXT = data.Field(batch_first=True, eos_token='<eos>')
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    if not text_flag:
        TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

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

    train_pos_iter, train_neg_iter = map(lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set])
    dev_pos_iter, dev_neg_iter = map(lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set])
    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)
    
    return train_iters, dev_iters, test_iters, vocab, TEXT

def load_dataset_mt(config, train_pos='train.en', train_neg='train.hi',
                 dev_pos='dev.en', dev_neg='dev.hi',
                 test_pos='test.en', test_neg='test.hi', train_pos2="train.pos", train_neg2="train.neg"):

    root = config.mt_data_path
    root2 = config.data_path

    TEXT = data.Field(batch_first=True, eos_token='<eos>')
    
    dataset_fn = lambda name: data.TabularDataset(
        path=root + name,
        format='tsv',
        fields=[('text', TEXT)]
    )
    dataset_fn2 = lambda name: data.TabularDataset(
        path=root2 + name,
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    train_pos_set2, train_neg_set2 = map(dataset_fn2, [train_pos2, train_neg2])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    TEXT.build_vocab(train_pos_set, train_neg_set, train_pos_set2, train_neg_set2, min_freq=config.min_freq)

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

    train_pos_iter, train_neg_iter = map(lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set])
    dev_pos_iter, dev_neg_iter = map(lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set])
    test_pos_iter, test_neg_iter = map(lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set])

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)
    
    return train_iters, dev_iters, test_iters, vocab, TEXT



# class Config():
#     data_path = './data/en-hi/'
#     log_dir = 'runs/exp'
#     save_path = './save'
#     pretrained_embed_path = './embedding/'
#     discriminator_method = 'Multi' # 'Multi' or 'Cond'
#     load_pretrained_embed = False
#     min_freq = 6
#     batch_size = 32
#     device = "cpu"
    
    
# if __name__ == '__main__':
#     config = Config()
#     train_iter, _, _, vocab = load_dataset_mt(config)
#     print(len(vocab))
#     for batch in train_iter:
#         text = tensor2text(vocab, batch.text)
#         print('\n'.join(text))
#         print(batch.label)
#         break


