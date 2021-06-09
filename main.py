import torch
import time
from data import load_dataset, load_dataset_mt
from models import StyleTransformer, Discriminator
from train import train, train_mt, auto_eval
from evaluate import evaluator


class Config():
    data_path = './data/cs-norepeat/'
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
    D_pretrain_iter = 1000
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0.1
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.5
    cyc_factor = 0.5
    adv_factor = 1.5
    mt_factor = 0.25

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
    
    ## for mt
    mt_data_path = './data/en-hi/'
    if_mt = True
    mt_steps = 200000
    pretrained_mt_model = None #"./save/MTJun07221615/ckpts/_F.pth"

    # for evaluate
    if_evaluate = False
    test_file = './data/cs-norepeat/used_for_gen'
    model_F_path = "./save/Jun08011647/ckpts/775_F.pth"
    model_D_path = "./save/Jun08011647/ckpts/775_D.pth"
    output_file = "./save/Jun08011647/ckpts/775_out.txt"
    
def to_train(config):
    if config.if_mt:
        train_iters_mt, dev_iters_mt, test_iters_mt, vocab, TEXT = load_dataset_mt(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        if config.pretrained_mt_model:
            print("Loading pretrained_mt_model")
            model_F.load_state_dict(torch.load(config.pretrained_mt_model))
        
        train_iters, dev_iters, test_iters, _, _ = load_dataset(config, TEXT)
        print('Vocab size:', len(vocab))
        model_D = Discriminator(config, vocab).to(config.device)
        train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters, train_iters_mt, dev_iters_mt, test_iters_mt)
    else:
        train_iters, dev_iters, test_iters, vocab, TEXT = load_dataset(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        model_D = Discriminator(config, vocab).to(config.device)
        print(config.discriminator_method)
        
        train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters, None, None, None)

def to_evaluate(config):
    if config.if_mt:
        train_iters, dev_iters, test_iters, vocab, TEXT = load_dataset_mt(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        model_F.load_state_dict(torch.load(config.model_F_path))
        evaluator(config, TEXT, model_F, config.test_file)

    else:
        train_iters, dev_iters, test_iters, vocab, TEXT = load_dataset(config)
        print('Vocab size:', len(vocab))
        model_F = StyleTransformer(config, vocab).to(config.device)
        model_F.load_state_dict(torch.load(config.model_F_path))
        print(config.discriminator_method)
        evaluator(config, TEXT, model_F, config.test_file)



def main():
    config = Config()
    if not config.if_evaluate:
        to_train(config)
    else:
        to_evaluate(config)

    

if __name__ == '__main__':
    main()
