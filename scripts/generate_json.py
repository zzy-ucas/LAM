# coding=utf-8
import os
import json
import random
import argparse


def main(args):
    dataset = args.dataset
    seed = args.seed
    # ratio = args.ratio
    generate_txt = args.generate_txt
    generate_path = args.generate_path
    random.seed(seed)

    if not os.path.isdir(generate_path):
        os.mkdir(generate_path)

    for dataset_type, path in dataset.items():
        with open(path, 'r') as f:
            data = json.load(f)

        total_data = data['images']
        train_list = list()
        val_list = list()
        test_list = list()
        # # 样本更加均衡，每一类别都抽出来10%用来验证
        if dataset_type == 'ucm' or dataset_type == 'sydney':
            for image in total_data:
                if image['split'] == 'train':
                    train_list.append(image)
                elif image['split'] == 'val':
                    val_list.append(image)
                else:
                    test_list.append(image)

        ########## RSICD ##########
        elif dataset_type == 'rsicd':
            classList = [
                'airport', 'bareland', 'baseballfield', 'beach', 'bridge',
                'center', 'church', 'commercial', 'denseresidential', 'desert',
                'farmland', 'forest', 'industrial', 'meadow', 'mediumresidential',
                'mountain', 'park', 'school', 'square', 'parking',
                'playground', 'pond', 'viaduct', 'port', 'railwaystation',
                'resort', 'river', 'sparseresidential', 'storagetanks', 'stadium',
            ]
            total_data_clean = list()
            totalData = {cls: list() for cls in classList}
            for image in total_data:
                if not str(image['filename']).startswith('00'):
                    total_data_clean.append(image)
            print('****', len(total_data_clean))

            for image in total_data_clean:
                ind = image['filename'].split('_')[0]
                totalData[ind].append(image)
            # total_data = total_data_clean


            for key, value in totalData.items():
                tmpList = totalData[key]
                random.shuffle(tmpList)
                train_list.extend(tmpList[:int(0.8 * len(tmpList))])
                val_list.extend(tmpList[int(0.8 * len(tmpList)):int(0.9 * len(tmpList))])
                test_list.extend(tmpList[int(0.9 * len(tmpList)):])

            for img in train_list:
                img['split'] = 'train'
            for img in val_list:
                img['split'] = 'val'
            for img in test_list:
                img['split'] = 'test'
            total_data = train_list + val_list + test_list
            data['images'] = total_data
            with open(path.replace('.json', '_1w.json'), 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        ########## RSICD ##########

        #####
        # random.shuffle(total_data)
        # train_list = total_data[:int(ratio * len(total_data))]
        # val_list = total_data[int(ratio * len(total_data)):]

        # random.shuffle(train_list)
        #### imgid:8734 in rsicd is wrong!
        dataset_list = {
            'total': total_data,
            'train': train_list,
            'val': val_list,
            'test': test_list
        }
        print('total_data:', len(total_data))
        print('train_list:', len(train_list))
        print('val_list:', len(val_list))
        print('test_list:', len(test_list))

        for phase, image_list in dataset_list.items():

            save_dict = dict()
            images = list()
            annotations = list()
            if generate_txt:
                imagenames = list()

            for img in image_list:
                image = dict()
                filename = img['filename']
                imgid = img['imgid']

                # image['file_name'] = filename
                image['file_name'] = filename.replace('.tif', '.jpg')
                image['id'] = imgid

                images.append(image)
                if generate_txt:
                    imagenames.append(filename)

                for num in range(5):
                    annotation = dict()
                    annotation['image_id'] = img['sentences'][num]['imgid']
                    annotation['id'] = img['sentences'][num]['sentid']
                    annotation['caption'] = img['sentences'][num]['raw']

                    annotations.append(annotation)

            # print(len(images))
            # print(len(annotations))
            save_dict['info'] = {u'description': u'This is stable 1.0 version of the 2014 MS COCO dataset.',
                                 u'url': u'http://mscoco.org', u'version': u'1.0', u'year': 2014,
                                 u'contributor': u'Microsoft COCO group',
                                 u'date_created': u'2015-01-27 09:11:52.357475'}
            save_dict['licenses'] = [{u'url': u'http://creativecommons.org/licenses/by-nc-sa/2.0/', u'id': 1,
                                      u'name': u'Attribution-NonCommercial-ShareAlike License'},
                                     {u'url': u'http://creativecommons.org/licenses/by-nc/2.0/', u'id': 2,
                                      u'name': u'Attribution-NonCommercial License'},
                                     {u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/', u'id': 3,
                                      u'name': u'Attribution-NonCommercial-NoDerivs License'},
                                     {u'url': u'http://creativecommons.org/licenses/by/2.0/', u'id': 4,
                                      u'name': u'Attribution License'},
                                     {u'url': u'http://creativecommons.org/licenses/by-sa/2.0/', u'id': 5,
                                      u'name': u'Attribution-ShareAlike License'},
                                     {u'url': u'http://creativecommons.org/licenses/by-nd/2.0/', u'id': 6,
                                      u'name': u'Attribution-NoDerivs License'},
                                     {u'url': u'http://flickr.com/commons/usage/', u'id': 7,
                                      u'name': u'No known copyright restrictions'},
                                     {u'url': u'http://www.usa.gov/copyright.shtml', u'id': 8,
                                      u'name': u'United States Government Work'}]
            save_dict['type'] = 'captions'
            save_dict['images'] = images
            save_dict['annotations'] = annotations

            target_path = os.path.join(
                generate_path, 'captions_%s_%s.json' % (dataset_type, phase))

            with open(target_path, 'w') as fp:
                json.dump(save_dict, fp, ensure_ascii=False, indent=4)

            if generate_txt:
                target_txt_path = os.path.join(
                    generate_path, 'captions_%s_%s.txt' % (dataset_type, phase))

                with open(target_txt_path, 'w') as fp:
                    for imagename in imagenames:
                        imagename = imagename.replace('.tif', '.jpg')
                        fp.write(imagename)
                        fp.write('\n')

            print(dataset_type, phase, 'done!')
    print('Done!')


if __name__ == '__main__':
    dataset = {
        # 'ucm': '/data/UCM_captions/dataset_jpg.json',
        # 'sydney': '/data/Sydney_captions/dataset_jpg.json',
        'rsicd': '/data/RSICD/dataset_jpg.json',
        # 'ucm': '/data/UCM_captions/dataset_non_error.json',
        # 'sydney': '/data/Sydney_captions/dataset_non_error.json',
    }

    parse = argparse.ArgumentParser()

    parse.add_argument('--dataset', type=dict, default=dataset)
    parse.add_argument('--seed', type=int, default=123)
    # parse.add_argument('--ratio', type=float, default=0.8)
    parse.add_argument('--generate_txt', type=bool, default=False)
    parse.add_argument('--generate_path', type=str, default='../annotations')

    args = parse.parse_args()

    print(args)
    main(args)
