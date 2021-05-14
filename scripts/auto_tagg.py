# coding:utf-8
import nltk
import json
from nltk.tag.stanford import StanfordPOSTagger
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np

tagger = StanfordPOSTagger(model_filename='/workspace/english-bidirectional-distsim.tagger',
                           path_to_jar='/workspace/stanford-postagger.jar')

dataset_path = '../annotations/captions_ucm_total.json'
semantic_path = '/data/UCM_captions/stanford_semantic_words.json'
# dataset_path = '../annotations/captions_sydney_total.json'
# semantic_path = '/data/Sydney_captions/stanford_semantic_words.json'
# dataset_path = '../annotations/captions_rsicd_total.json'
# semantic_path = '/data/RSICD/stanford_semantic_words.json'

semantic_dict = dict()
# tag_dict = dict()
# count = Counter()
# with open(dataset_path, 'r') as f:
#     data = json.load(f)
#     print(data['dataset'])
#     for img in data['images']:
#         imgid = img['imgid']
#         # semantic_list = []
#         for sent in img['sentences']:
#             # print(sent['tokens'])
#             # semantic_list.append(sent['tokens'])
#             # if '' in sent['tokens']:
#             #     sent['tokens']=sent['tokens'].remove('')
#             # if sent['tokens'] is None:
#             #     continue
#             count.update(sent['raw'].lower().split())
# # vocab = count.keys()
# vocab = [word for word, cnt in count.items() if cnt >= 5]

coco = COCO(dataset_path)
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if i % 1000 == 0:
        print("[%d/%d] Tokenized the captions." % (i, len(ids)))

#####
# cw = sorted([(count, w) for w, count in counter.items()], reverse=True)
# print('\n'.join(map(str, cw[:20])))
# with open('/data/RSICD/tmp.json', 'w') as f:
#     json.dump(cw, f)
# fig, ax = plt.subplots()
# positions = np.arange(1, len(cw)+1)
# heights = [i[1] for i in cw]
# ax.bar(positions, heights, 0.5)
# plt.savefig('figs/ucm.jpg')
#####
# If the word frequency is less than 'threshold', then the word is discarded.
vocab = [word for word, cnt in counter.items() if cnt >= 5]

"""
数据处理规则:
1. nltk.tag
2. remove numbers. eg: 'three'.
3. remove ambiguous noun. eg: 'color' 'corner'.
4. remove verb. eg: 'go' 'compose' 'roadside' 'side' 'positions' 'corners'
5. remove agj. eg: 'compact' 'others'
"""
filter_list_ucm = [
    'parrallel',
    'color',
    'go',
    'corner',
    'compact',
    'one',
    'shadow',
    'size',
    'position',
    'side',
    'row',
    'other',
    'direction',
    'quiet',
    'swimming',
    'flying',
    'number',
    'area',
    'lot',
    'line',
    'nothing',
    'kind',
    'cross',
    'well',
    'piece',
    'waste',
    'turbid',
    'beat',
    'atrovirens',
    'part',
    'crystal',
    'slapping',
    'mess',
    'l',
    'r',
    'space',
    'home',
    'ground',
    'spot',
]
filter_list_sydney = [
    'distance',
    'float',
    'main',
    'pass',
    'middle',
    'angle',
    'flow',
    'wave',
    'rectangle',
    'curving',
    'complex',
]
for i in filter_list_ucm:
    if i in vocab:
        vocab.remove(i)
    if i + 's' in vocab:
        vocab.remove(i + 's')
    if i + 'es' in vocab:
        vocab.remove(i + 'es')
    if i + 'ing' in vocab:
        vocab.remove(i + 'ing')
for i in filter_list_sydney:
    if i in vocab:
        vocab.remove(i)
    if i + 's' in vocab:
        vocab.remove(i + 's')
    if i + 'es' in vocab:
        vocab.remove(i + 'es')
    if i + 'ing' in vocab:
        vocab.remove(i + 'ing')

for word, tag in tagger.tag(vocab):
    if tag.startswith('N'):
        semantic_dict[word] = semantic_dict.get(word, 0) + 1
        # tag_dict[word] = tag
# print(semantic_dict.keys())
with open(semantic_path, 'w') as fp:
    json.dump(semantic_dict.keys(), fp, ensure_ascii=False, indent=4)
