import os
import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import re


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold, category=None):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    meanLen = 0.0
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        # caption = re.sub("[\.\!,]+", "", caption)  # remove "."
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        meanLen += float(len(tokens)) / float(len(ids))
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(ids)))
    print("mean length captions: ", meanLen)
    # cw = sorted([(count, w) for w, count in counter.items() if count >= threshold], reverse=True)
    # print('********************')
    # print('\n'.join(map(str, cw)))
    # print('********************')
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add category sI4-sI24
    if category is not None:
        for i in sorted(category):
            vocab.add_word(i.lower())

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    if not os.path.exists('data'):
        os.mkdir('data')

    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold,
                        category=args.category
                        )
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" % len(vocab))  # 299 for UCM, 185 for Sydney
    print("Saved the vocabulary wrapper to '%s'" % vocab_path)


if __name__ == '__main__':
    caption_paths = {
        # 'ucm': './annotations/captions_ucm_total.json',
        # 'sydney': './annotations/captions_sydney_total.json',
        # 'rsicd': './annotations/captions_rsicd_total.json',
        'ucm': './annotations/captions_ucm_train.json',
        'sydney': './annotations/captions_sydney_train.json',
        'rsicd': './annotations/captions_rsicd_train.json',
    }

    categories = {
        'ucm': ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
                'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
                'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
                'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks',
                'tenniscourt'],
        'sydney': ['Residential', 'Airport', 'Meadow', 'River', 'Ocean', 'Industrial', 'Runway'],
        'rsicd': ['airport', 'denseresidential', 'park', 'school', 'bareland',
                  'desert', 'parking', 'sparseresidential', 'baseballfield', 'farmland',
                  'playground', 'square', 'beach', 'forest', 'pond',
                  'stadium', 'bridge', 'industrial', 'port', 'storagetanks',
                  'center', 'meadow', 'railwaystation', 'viaduct', 'church',
                  'mediumresidential', 'resort', 'commercial', 'mountain', 'river', ]
    }

    # current_dataset = 'ucm'
    # current_dataset = 'sydney'
    current_dataset = 'rsicd'

    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default=caption_paths[current_dataset],
                        help='path for train annotation file')
    parser.add_argument('--category', type=list,
                        default=categories[current_dataset],
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_%s.pkl' % current_dataset,
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
