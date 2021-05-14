import os
# from skimage.io import imread, imsave
from scipy.misc import imread, imsave

path_src = [
    '/data/UCM_captions/imgs/',
    '/data/Sydney_captions/imgs/',
        ]
path_tar = [
    '/data/UCM_captions/jpg/',
    '/data/Sydney_captions/jpg/',
            ]

for idx in range(len(path_src)):
    src = path_src[idx]
    tar = path_tar[idx]

    if not os.path.exists(tar):
        os.mkdir(tar)

    for image in os.listdir(src):
        image_path = os.path.join(src, image)
        save_path = os.path.join(tar, image.replace('.tif', '.jpg'))

        img = imread(image_path)
        imsave(save_path, img)
