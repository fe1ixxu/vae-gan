import torch
import time
from data import load_dataset, load_dataset_mt
from models import StyleTransformer, Discriminator
from train import train, train_mt, auto_eval


class Config():
    data_path = './data/en-hi/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 90
    max_length = 64
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 5 #10
    iter_F = 30
    F_pretrain_iter = 1000
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 2

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0.15
    
    ## for mt
    if_mt = True
    mt_steps = 30
    


def main():
    config = Config()
    if config.if_mt:
        train_iters, dev_iters, test_iters, vocab = load_dataset_mt(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        train_mt(config, vocab, model_F, train_iters, dev_iters, test_iters)
        
        print("MT training finished")
        train_iters, dev_iters, test_iters, _ = load_dataset(config)
        print('Vocab size:', len(vocab))
        model_D = Discriminator(config, vocab).to(config.device)
        train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    else:
        train_iters, dev_iters, test_iters, vocab = load_dataset(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        model_D = Discriminator(config, vocab).to(config.device)
        print(config.discriminator_method)
        
        train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
