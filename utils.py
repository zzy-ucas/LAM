import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pickle
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms, datasets
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
from PIL import Image


# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# MS COCO evaluation data loader
class CocoImageFolder(datasets.ImageFolder):
    def __init__(self, root, ann_path, transform=None, target_transform=None, split=None,
                 loader=datasets.folder.default_loader):
        """
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        """
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        self._loader = loader
        self._split = split
        self._images = json.load(open(ann_path, 'r'))['images']

        # dict to convert image filename to image id in COCO
        # self.image_ids_to_fileNames = {image['id']: image['filename'] for image in self._images}
        self._build_index()

    def _build_index(self):
        eval_imgs = list()

        for image in self._images:
            if self._split is 'val' and image['split'] == 'val':
                eval_imgs.append(image)
            elif self._split is 'test' and image['split'] == 'test':
                eval_imgs.append(image)

        self._eval_imgs = eval_imgs
        print('Evaluating images:', len(self._eval_imgs))

    def __getitem__(self, index):
        # Filename for the image
        img = self._eval_imgs[index]
        imgid = img['imgid']
        filename = img['filename']

        image = Image.open(os.path.join(self._root, filename)).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        if self._transform is not None:
            image = self._transform(image)

        return image, imgid, filename

    def __len__(self):
        return len(self._eval_imgs)


# MSCOCO Evaluation function
def coco_eval(model, args, epoch, split=None):
    '''
    model: trained model to be evaluated
    args: pre-set parameters
    epoch: epoch #, for disp purpose
    '''

    model.eval()

    # Validation images are required to be resized to 224x224 already
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load the vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Wrapper the COCO VAL dataset
    eval_data_loader = torch.utils.data.DataLoader(
        CocoImageFolder(args.image_dir, args.caption_path, transform, split=split),
        batch_size=args.eval_size,
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)

    # Generated captions to be compared with GT
    results = []
    print('---------------------Start evaluation on MS-COCO dataset %s-----------------------' % split)
    for i, (images, image_ids, filename) in enumerate(eval_data_loader):
        images = to_var(images)

        if torch.cuda.device_count()>1:
            device_ids = range(torch.cuda.device_count())
            encoder_parallel = nn.DataParallel(model.encoder, device_ids=device_ids)
            features, probs = encoder_parallel(images)
        else:
            features, probs = model.encoder(images)
        if args.pattern == 'truelabel':
            if args.dataset == 'ucm' or args.dataset == 'sydney':
                preds = torch.LongTensor(image_ids) // 100
                preds += 4
                preds = to_var(preds).unsqueeze(1)
            elif args.dataset == 'rsicd':
                trueLabels = [args.vocab(str(fn).split('_')[0]) for fn in filename]
                trueLabels = torch.LongTensor(trueLabels)
                preds = to_var(trueLabels).unsqueeze(1)
        elif args.pattern == 'label':
            preds = torch.max(probs.data, 1)[1].unsqueeze(1)
            preds += 4
            preds = to_var(preds)
        else:
            preds = None

        # generated_captions, _ = model.decoder.sample(features)
        generated_captions, _ = model.decoder.sample(features, preds, args.pattern)  #sat(adaptive)
        # generated_captions, _ = model.decoder.sample(features, probs, args.pattern)  #fc_lstm

        captions = generated_captions.cpu().data.numpy()

        # Build caption based on Vocabulary and the '<end>' token
        for image_idx in range(captions.shape[0]):

            sampled_ids = captions[image_idx]
            sampled_caption = []

            for word_id in sampled_ids:

                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)

            sentence = ' '.join(sampled_caption)

            temp = {'image_id': int(image_ids[image_idx]), 'caption': sentence}
            results.append(temp)

        # Disp evaluation process
        if (i + 1) % 100 == 0:
            print('[%d/%d]' % ((i + 1), len(eval_data_loader)))

    print('------------------------Caption Generated-------------------------------------')

    # Evaluate the results based on the COCO API
    # name = str(args.yml).split('.')[0].split('/')[-1]
    if not os.path.exists(os.path.join(args.checkpoint_path, 'results')):
        os.mkdir(os.path.join(args.checkpoint_path, 'results'))
    if split is 'val':
        resFile = os.path.join(args.checkpoint_path, 'results',
                               args.dataset + '-' + '{0:03d}'.format(epoch) + '.json')
    else:
        resFile = os.path.join(args.checkpoint_path, 'results',
                               args.dataset + '-' + split + '-{0:03d}'.format(epoch) + '.json')
        # resFile = os.path.join(args.checkpoint_path,
        #                        args.dataset + '-' + split + '.json')
    json.dump(results, open(resFile, 'w'), indent=4)

    annFile = args.caption_val_path
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Get CIDEr score for validation evaluation
    cider = 0.
    print('-----------Evaluation performance on MS-COCO validation dataset for Epoch %d----------' % (epoch))
    for metric, score in cocoEval.eval.items():

        print('%s: %.4f' % (metric, score))
        if metric == 'CIDEr':
            cider = score

    return cider, cocoEval.eval
