import json

dataset_path = '/data/UCM_captions/dataset_jpg.json'
semantic_path = '/data/UCM_captions/semantic_words.json'

semantic_dict = dict()
with open(dataset_path, 'r') as f:
    data = json.load(f)
    print(data['dataset'])
    for img in data['images']:
        imgid = img['imgid']
        # semantic_list = []
        for sent in img['sentences']:
            # print(sent['tokens'])
            # semantic_list.append(sent['tokens'])
            for word in sent['tokens']:
                semantic_dict[word] = semantic_dict.get(word, 0) + 1

with open(semantic_path, 'w') as fp:
    json.dump(semantic_dict, fp, ensure_ascii=False, indent=4)

# semantic_words_sin='/data/Sydney_captions/semantic_singular.json'
# semantic_words_txt='/data/Sydney_captions/semantic_singular.json'
# with open(semantic_words_sin, 'r') as f:
#     data=json.load(f)
#     semantcis=list(data.keys())
# with open(semantic_words_txt, 'w') as fp:
#     json.dump(semantcis, fp)
#