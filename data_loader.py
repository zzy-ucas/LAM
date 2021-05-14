import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random
import json
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

# random.seed(123)
# print('Setting shuffle seed.')


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, ann_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            ann_path: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self._root = root
        self._annotations = json.load(open(ann_path))
        self._vocab = vocab
        self._transform = transform
        self._build_index()

    def _build_index(self):
        train_imgs = list()

        for image in self._annotations['images']:
            if image['split'] == 'train':
                train_imgs.append(image)
        # random.shuffle(train_imgs)
        self._train_imgs = train_imgs
        print('Training images:', len(self._train_imgs))

    def __getitem__(self, index):
        """Returns one data pair (image, caption, image_id)."""
        vocab = self._vocab
        img = self._train_imgs[index]
        file_name = img['filename']

        image = Image.open(os.path.join(self._root, file_name)).convert('RGB')
        # image = image.resize([224, 224], Image.LANCZOS)
        if self._transform is not None:
            image = self._transform(image)
        images = torch.stack([image] * 5, dim=0).squeeze(0)
        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        captions = []
        for cap in img['sentences']:
            caption = []
            caption.append(vocab('<start>'))
            tokens = nltk.tokenize.word_tokenize(str(cap['raw']).lower())
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            captions.append(caption)

        return images, captions

    def __len__(self):
        return len(self._train_imgs)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
    """
    images, captions = zip(*data)  # unzip
    images = torch.cat(images, 0)
    captions = reduce(lambda x, y: x + y, captions)

    # Sort by caption length
    images, captions = zip(*sorted(zip(images, captions), key=lambda x: len(x[1]), reverse=True))  # unzip

    # Convert images from Tuple to Tensor
    images = [_ for _ in images]
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])

    return images, targets, lengths


def get_loader(root, ann_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       ann_path=ann_path,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
