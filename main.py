import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import tensorboard
import torch
from tensorboardX import SummaryWriter

from trainer import run_training
from test import run_test

parser = argparse.ArgumentParser()

## model
parser.add_argument('--model_name', default='transformer', type=str, help='cnn | transformer')
## optimization
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm adam | adamw | sgd')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
parser.add_argument('--max_epochs', default=100, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--es', default=50, type=int)

## data
# parser.add_argument('--data_dir', default='/media/yinn147/Data/ICC_new/new_data/crop_tumorsize_x15/', type=str, help='dataset directory')

parser.add_argument('--train_json_list', default='/media/yinn147/Data/ICC_transformer/preprocess/json/trainval.json', type=str, help='dataset json file')
parser.add_argument('--intest_json_list', default='/media/yinn147/Data/ICC_transformer/preprocess/json/internaltest.json', type=str, help='dataset json file')
parser.add_argument('--extest_json_list', default='/media/yinn147/Data/ICC_transformer/preprocess/json/externaltest.json', type=str, help='dataset json file')

parser.add_argument('--mods', default=0, type=int, help='single modality 0/1/2/3/4/5/6 for A, ADC, D, T1PRE, DWI, T2, V')
parser.add_argument('--multi_use', action='store_true', help='use multi-modalities or not')
parser.add_argument('--resample', nargs = '+', type=int, default=[64, 64, 64], help='size of input image')
parser.add_argument('--in_channels', default=7, type=int, help='number of input channels (if multi_use : 7, if single mod : 1)')

## log
parser.add_argument('--pretrained_dir', default='', type=str, help='pretrained checkpoint directory')
parser.add_argument('--logs', default='/media/yinn147/Data/ICC_transformer/log_new/', type=str, help='logging path (dont forget / at last)')
parser.add_argument('--note', default='_test_', type=str, help='experiment setting notification')

### evalutaion 
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_indicator', default='auc', type=str, help='acc | auc')

## other
parser.add_argument('--feature_size', default=48, type=int, help='feature size')
parser.add_argument('--out_channels', default=1, type=int, help='number of output channels')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--val_every', default=1, type=int, help='validation frequency')

def seed_torch(seed=45):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(42)
    args = parser.parse_args()

    # args.multi_use= True
    # args.in_channels = 7
    # args.resample=[32, 64, 64]
    # args.test=True
    # args.pretrained_dir = '/media/yinn147/Data/ICC_transformer/log_new/multimod_transformer_326464_2024-09-03T23:37:34'

    ############### log ##################
    if args.test:
        args.logdir = f'{args.pretrained_dir}/test_best{args.test_indicator}_{datetime.datetime.now().isoformat()[:19]}/'
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.FileHandler(f'{args.logdir}test.log'), logging.StreamHandler(sys.stdout)])
        logging.info(str(args))
        logging.info('Testing for ' + args.pretrained_dir)

        run_test(args)
    else:
        if args.multi_use:
            args.logdir = f'{args.logs}multimod_{args.model_name}_{args.resample[0]}{args.resample[1]}{args.resample[2]}_{datetime.datetime.now().isoformat()[:19]}/'
        else:
            args.logdir = f'{args.logs}singlemod_{args.mods}_{args.model_name}_{args.resample[0]}{args.resample[1]}{args.resample[2]}_{datetime.datetime.now().isoformat()[:19]}/'
        
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        writer = SummaryWriter(log_dir=args.logdir)
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.FileHandler(f'{args.logdir}train.log'), logging.StreamHandler(sys.stdout)])
        logging.info(str(args))

        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
    
        run_training(args=args, tensorboard_writer = writer)

if __name__ == '__main__':
    main()
