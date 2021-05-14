import os
import math
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from utils import coco_eval, to_var
from data_loader import get_loader
from adaptive import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tensorboardX import SummaryWriter
import cPickle


# def add_summary_value(writer, tag, value, iteration):
#     if writer:
#         writer.add_scalar(tag, value, iteration)
#
#
# def add_summary_dict(writer, tag, kv, iteration):
#     if writer:
#         writer.add_scalars(tag, kv, iteration)


# print('###', torch.cuda.device_count())  # the number of GPU is 16, but we only use one.


def main(args):
    # tb_summary_writer = SummaryWriter(args.checkpoint_path)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True

    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args, len(vocab))
    if torch.cuda.is_available():
        adaptive.cuda()
    # adaptive = Encoder2Decoder(args, len(vocab), args.gpu)
    if vars(args).get('start_from', None) is not None and os.path.isfile(args.start_from):
        adaptive.load_state_dict(torch.load(args.start_from))
    # cider_scores = []

    # Start Training
    # for epoch in range(start_epoch, args.num_epochs + 1):

    cider, metrics = coco_eval(adaptive, args, 0, split='test')
    print('Testing Model: CIDEr score %.2f' % (cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #####
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--yml', type=str, default='config/sydney_test.yml')
    # parser.add_argument('--use_class', type=bool, default=False)
    # parser.add_argument('--use_true_label', type=bool, default=False)
    # parser.add_argument('--vgg_model_path', type=str, default='')
    # parser.add_argument('--num_classes', type=int)
    parser.add_argument('--image_dir', type=str, help='directory for resized training images')
    parser.add_argument('--caption_path', type=str, help='path for train annotation json file')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--caption_val_path', type=str, help='path for validation annotation json file')
    parser.add_argument('--start_from', type=str, default=None)
    #####
    parser.add_argument('--model_path', type=str, default='./models-attentive/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,  # default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    # parser.add_argument('--val_dir', type=str, default='./data/resized/val2014',
    #                     help='directory for resized validation images')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------

    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=6,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=10,
                        help='start fine-tuning CNN after')

    # Optimizer Adam parameter
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in Adam')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='beta in Adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='learning rate for the whole model')
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4,
                        help='learning rate for fine-tuning CNN')

    # LSTM hyper parameters
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='dimension of word embedding vectors, also dimension of v_g')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--vis_dim', type=int, default=512)
    parser.add_argument('--vis_num', type=int, default=196)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    # Training details
    # parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    # parser.add_argument('--pretrained', type=str, default='models-attentive/adaptive-48.pkl', help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=500)
    # parser.add_argument('--batch_size', type=int, default=50)

    # For eval_size > 30, it will cause cuda OOM error.
    parser.add_argument('--eval_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=20, help='epoch at which to start lr decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')

    args = parser.parse_args()

    import yaml

    print(args.yml)
    with open(args.yml, 'r') as f:
        data = yaml.load(f)
        for k, v in data.items():
            if k in args:
                setattr(args, k, v)

    print('------------------------Model and Training Details--------------------------')
    print(args)

    # Start training
    main(args)
