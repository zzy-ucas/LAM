import os
import math
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from utils import coco_eval, to_var
from data_loader import get_loader
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from tensorboardX import SummaryWriter
import cPickle

# from fc_lstm import Encoder2Decoder
from adaptive import Encoder2Decoder


def add_summary_value(writer, tag, value, iteration):
    if writer:
        writer.add_scalar(tag, value, iteration)


def add_summary_dict(writer, tag, kv, iteration):
    if writer:
        writer.add_scalars(tag, kv, iteration)


# print('###', torch.cuda.device_count())  # the number of GPU is 16, but we only use one.


def main(args):
    args.checkpoint_path = os.path.join('log_' + args.dataset + '_' + args.pattern, args.session)
    tb_summary_writer = SummaryWriter(args.checkpoint_path)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True

    # To reproduce training results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print('### CUDA is available!')
        torch.cuda.manual_seed(args.seed)

    # Create model directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists('data'):
        os.mkdir('data')

    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        args.vocab = vocab

    # Build training data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder(args, len(vocab))
    # adaptive = Encoder2Decoder(args, len(vocab), args.gpu)

    infos = {}
    if args.start_from is not None:
        with open(os.path.join(args.start_from, 'infos_' + args.dataset + '.pkl')) as f:
            infos = cPickle.load(f)
            # saved_model_opt = infos['args']
            # need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            # for checkme in need_be_same:
            #     assert vars(saved_model_opt)[checkme] == vars(args)[
            #         checkme], "Command line argument and saved model disagree on '%s' " % checkme
    if vars(args).get('start_from', None) is not None and os.path.isfile(
            os.path.join(args.start_from, "model.pth")):
        adaptive.load_state_dict(torch.load(os.path.join(args.start_from, 'model.pth')))

    epoch = infos.get('epoch', 1)

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list(adaptive.encoder.vgg_conv.children())
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn,
                                     betas=(args.alpha, args.beta))
    if vars(args).get('start_from', None) is not None and os.path.isfile(
            os.path.join(args.start_from, "cnn_optimizer.pth")):
        cnn_optimizer.load_state_dict(torch.load(os.path.join(args.start_from, 'cnn_optimizer.pth')))

    # Other parameter optimization
    params = list(adaptive.decoder.parameters())

    # Will decay later    
    learning_rate = args.learning_rate

    # Language Modeling Loss, Optimizers
    LMcriterion = nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    # Train the Models
    total_step = len(data_loader)

    cider_scores = []
    best_cider = 0.0
    best_epoch = 0
    best_cider_test = 0.0
    best_epoch_test = 0
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    if vars(args).get('start_from', None) is not None and os.path.isfile(
            os.path.join(args.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(args.start_from, 'optimizer.pth')))

    # Start Training 
    # for epoch in range(start_epoch, args.num_epochs + 1):
    update_lr_flag = True
    while True:
        if update_lr_flag:
            if epoch > args.lr_decay:
                frac = (epoch - args.cnn_epoch) / args.learning_rate_decay_every
                decay_factor = math.pow(0.5, frac)

                # Decay the learning rate
                learning_rate = learning_rate * decay_factor
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
            update_lr_flag = False
        # Language Modeling Training
        print('------------------Training for Epoch %d----------------' % (epoch))
        cur_time = time.time()
        for i, (images, captions, lengths) in enumerate(data_loader):
            start_time = time.time()
            # print('### images:', images.size())
            # print('### captions:', captions.size())
            # print('### lengths:', len(lengths))
            # Set mini-batch dataset
            images = to_var(images)
            captions = to_var(captions)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            # Forward, Backward and Optimize
            adaptive.train()
            adaptive.zero_grad()

            packed_scores = adaptive(images, captions, lengths, args.pattern)

            # Compute loss and backprop
            loss = LMcriterion(packed_scores[0], targets)
            loss.backward()

            # Gradient clipping for gradient exploding problem in LSTM
            for p in adaptive.decoder.lstm_cell.parameters():
                p.data.clamp_(-args.clip, args.clip)

            optimizer.step()

            # Start learning rate decay

            # Start CNN fine-tuning
            if epoch > args.cnn_epoch:
                cnn_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f, Elapsed: %.2fs' % \
                      (epoch, args.num_epochs,
                       i, total_step,
                       loss.item(),
                       np.exp(loss.item()),
                       time.time() - start_time))

                add_summary_value(tb_summary_writer, 'loss', loss.item(), epoch)
        print('##### Per Epoch Cost time: %.2fs' % (time.time() - cur_time))
        infos['epoch'] = epoch
        infos['vocab'] = vocab
        infos['args'] = args
        with open(os.path.join(args.checkpoint_path, 'infos.pkl'), 'wb') as f:
            cPickle.dump(infos, f)
        torch.save(optimizer.state_dict(),
                   os.path.join(args.checkpoint_path, 'optimizer.pth'))
        torch.save(cnn_optimizer.state_dict(),
                   os.path.join(args.checkpoint_path, 'cnn_optimizer.pth'))
        torch.save(adaptive.state_dict(),
                   os.path.join(args.checkpoint_path, 'model.pkl'))
        # with open(os.path.join(args.checkpoint_path, 'histories.pkl'), 'wb') as f:
        #     cPickle.dump(infos, f)
        # Evaluation on validation set
        cider, metrics = coco_eval(adaptive, args, epoch, split='val')
        cider_scores.append(cider)
        add_summary_dict(tb_summary_writer, 'metrics', metrics, epoch)

        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch

            # Save the Adaptive Attention model after each epoch
            # name = str(args.yml).split('.')[0].split('/')[-1]
            torch.save(adaptive.state_dict(),
                       os.path.join(args.checkpoint_path, 'model-best.pkl'))
            with open(os.path.join(args.checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
        print('Model of best epoch #: %d with CIDEr score %.2f' % (best_epoch, best_cider))

        # Test on test set
        caption_val_path = args.caption_val_path
        args.caption_val_path = args.caption_val_path.replace('val', 'test')
        cider_test, metrics_test = coco_eval(adaptive, args, epoch, split='test')
        args.caption_val_path = caption_val_path
        if cider_test > best_cider_test:
            best_cider_test = cider_test
            best_epoch_test = epoch
        print('Test Phase: Model of best epoch #: %d with CIDEr score %.2f' % (best_epoch_test, best_cider_test))

        epoch += 1
        if epoch > 80:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #####
    parser.add_argument('--gpu', type=str, default='7')
    parser.add_argument('--dataset', type=str, default='ucm')
    parser.add_argument('--yml', type=str, default='config/ucm_truelabel_vgg11.yml')
    parser.add_argument('--session', type=str, default='000')
    parser.add_argument('--pattern', type=str, default='baseline')
    parser.add_argument('--vgg_model_path', type=str, default='')
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--image_dir', type=str, help='directory for resized training images')
    parser.add_argument('--caption_path', type=str, help='path for train annotation json file')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--caption_val_path', type=str, help='path for validation annotation json file')
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--vocab_path', type=str,  # default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    #####
    # parser.add_argument('--model_path', type=str, default='./checkpoints/',
    #                     help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    # parser.add_argument('--val_dir', type=str, default='./data/resized/val2014',
    #                     help='directory for resized validation images')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')

    # ---------------------------Hyper Parameter Setup------------------------------------

    # CNN fine-tuning
    # parser.add_argument('--fine_tune_start_layer', type=int, default=6,
    #                     help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
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
    # parser.add_argument('--vis_num', type=int, default=49)
    parser.add_argument('--vis_num', type=int, default=196)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    # Training details
    parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=50)

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
    import datetime

    print("### Start time: " + str(datetime.datetime.now()))
    main(args)
    print("### End time: " + str(datetime.datetime.now()))
