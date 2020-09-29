#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch
import glob

N_IMAGES = 202599
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pth'
FOLDER = '~/rendering_materials/renders/renders_materials_manu/'
FOLDER = '../../dataset/renders_by_geom_ldr/'
def preprocess_images():

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    attr_lines = [line.rstrip() for line in open(FOLDER+'attributes_dataset.txt', 'r')]
    N_IMAGES=len(attr_lines)-1

    imgs=[FOLDER+'256px_dataset/'+line.split('\t',1)[0] for line in attr_lines[1:]]


    # print("Reading images from img_align_celeba/ ...")
    # imgs=glob.glob(FOLDER+'/256px_dataset/*')
    N_IMAGES=len(imgs)
    print(N_IMAGES)
    raw_images = []
    for i in range(0, N_IMAGES):
        if i % 10000 == 0:
            print(i)
        raw_images.append(mpimg.imread(imgs[i]))

    if len(raw_images) != N_IMAGES:
        raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))

    print("Resizing images ...")
    all_images = []
    for i, image in enumerate(raw_images):
        if i % 10000 == 0:
            print(i)
        #assert image.shape == (178, 178, 3)
        if IMG_SIZE < 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        all_images.append(image)

    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    data = torch.from_numpy(data)
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)


def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = [line.rstrip() for line in open(FOLDER+'attributes_dataset.txt', 'r')]
    print(len(attr_lines))
    #assert len(attr_lines) == N_IMAGES + 1

    attr_keys = attr_lines[0].split()
    print (attr_keys)
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[1:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == len(attr_keys)+1
        #assert split[0] == ('%06i.jpg' % image_id)
        #assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = float(value)*2-1

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


preprocess_images()
preprocess_attributes()
