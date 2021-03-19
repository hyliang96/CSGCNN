#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
"""
preprocessing for Pascal VOC 2010 Part
crop each anminal instance in Pascal VOC 2020 and resize the image to size x size pixel

    number of processed instances = {'bird': 978, 'cat': 1132, 'dog': 1416, 'cow': 480, 'horse': 624, 'sheep': 714}
    number of raw images = 10103
    number of processed instances = 5344
    number of processed body parts = 50723


    ./__data__/processed/
        128x128/                               or 32x32/ 64x64/
            obj_img/<image-id>.jpg                3 channel, 8 bit, cropped and resized image of an instance
            obj_mask/<image-id>.bmp               1 channel, 8 bit, cropped and resized segmentation ground truth of an instance
            part_mask/<image-id>/<body-part>.bmp   1 channel, 8 bit, cropped and resized segmentation ground truth of each body parts of an instance
        metadata/
            train.txt    7072 lines i.e. images
            val.txt      3031 lines i.e. images
"""

# crop each anminal instance in Pascal VOC 2020 and resize the image to size x size
size = 128


# %%
import argparse
import PIL
import os
import matplotlib.pyplot as plt
import numpy as np
import utils


# Plot a mask composed by 0s and 1s with a certain title
# and compare it with the original image:
def plot_mask(img, mask, bodypart_mask, windowtitle, suptitle):
    img = PIL.Image.fromarray(img)
    mask = PIL.Image.fromarray(mask * 255)
    bodypart_mask = PIL.Image.fromarray(bodypart_mask * 255)
    fig = plt.figure()
    fig.canvas.set_window_title(windowtitle)
    fig.suptitle(suptitle)
    fig.add_subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(mask)
    fig.add_subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(bodypart_mask)
    plt.show()




def crop(img, obj_mask, part_mask=None):
    y_sum = (obj_mask.sum(axis=1)>0)
    y_max, y_min = int(np.argwhere(y_sum).max()), int(np.argwhere(y_sum).min())
    y_d   = y_max - y_min
    y_0   = (y_max + y_min)/2

    x_sum = (obj_mask.sum(axis=0)>0)
    x_max, x_min = int(np.argwhere(x_sum).max()), int(np.argwhere(x_sum).min())
    x_d   = x_max - x_min
    x_0   = (x_max + x_min)/2

    wid   = max(x_d, y_d)/2
    Y, X, channel  = img.shape

    def crop_img(I):
        I_pad = np.pad(I, ((Y,Y), (X,X),(0,0)), 'constant') # "symmetric" 'constant'
        I_crop = I_pad[round(Y+y_0-wid):round(Y+y_0+wid), round(X+x_0-wid):round(X+x_0+wid)]
        return I_crop

    def crop_mask(I):
        I_pad = np.pad(I, ((Y,Y), (X,X)), 'constant') # 'constant' zero_pad
        I_crop = I_pad[round(Y+y_0-wid):round(Y+y_0+wid), round(X+x_0-wid):round(X+x_0+wid)]
        return I_crop

    if part_mask is None:
        return crop_img(img), crop_mask(obj_mask)
    else:
        return crop_img(img), crop_mask(obj_mask), crop_mask(part_mask)

# Plot a mask composed by 0s and 1s with a certain title
# and compare it with the original image:
def plot_mask(img, mask, bodypart_mask, windowtitle, suptitle):
    img = PIL.Image.fromarray(img)
    mask = PIL.Image.fromarray(mask * 255)
    bodypart_mask = PIL.Image.fromarray(bodypart_mask * 255)

    fig = plt.figure()
    fig.canvas.set_window_title(windowtitle)
    fig.suptitle(suptitle)
    fig.add_subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(img)
    fig.add_subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(mask)
    fig.add_subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(bodypart_mask)
    plt.show()

# from scipy.misc import imresize
save_size = (size,size)

def np_save_resize_img(img, filename):
    # imresize(img, save_size)
    if img.max() == 1:
        img = PIL.Image.fromarray(img*255).convert('1')
        img = img.resize(save_size, resample=PIL.Image.NEAREST)
        img.save(filename + '.bmp')

    else:
        img = PIL.Image.fromarray(img)
        img = img.resize(save_size, resample=PIL.Image.LANCZOS)
        img.save(filename + '.jpg')

# %%

Part_dict = {
    'bird': {
        'HEAD':  [ 'head', 'reye', 'leye', 'beak'] ,
        'TORSO': ['torso', 'neck', 'lwing', 'rwing'],
        # 'FLIMB':  ['lwing', 'rwing'],
        'BLIMB': ['lleg', 'rleg', 'lfoot', 'rfoot'],
        'TAIL': ['tail']
    },

    'cat':{
        'HEAD': [ 'head',  'leye', 'reye', 'lear', 'rear', 'nose'],
        'TORSO': [ 'torso', 'neck'],
        'FLIMB': [ 'lfleg', 'rfleg', 'lfpa', 'rfpa'],
        'BLIMB': ['lbleg', 'rbleg', 'lbpa', 'rbpa'],
        'TAIL': ['tail']
    },

    'dog':{
        'HEAD': [ 'head', 'leye', 'reye', 'lear', 'rear',  'nose',  'muzzle'],
        'TORSO': ['torso', 'neck'],
        'FLIMB': ['lfleg',  'rfleg', 'lfpa', 'rfpa'],
        'BLIMB': ['rbleg',  'lbleg', 'rbpa', 'lbpa'],
        'TAIL': ['tail']
    },

    'cow':{
        'HEAD': ['head', 'lear', 'rear', 'leye', 'reye', 'muzzle',  'lhorn', 'rhorn'],
        'TORSO': ['torso', 'neck'],
        'FLIMB': ['lflleg', 'lfuleg', 'rflleg', 'rfuleg'],
        'BLIMB': ['lblleg', 'lbuleg', 'rblleg', 'rbuleg'],
        # 'TAIL': ['tail']
    },

    'horse':{
        'HEAD': ['head', 'lear', 'rear','leye', 'reye',  'muzzle'],
        'TORSO': ['torso', 'neck'],
        'FLIMB': ['lflleg', 'lfuleg', 'rflleg', 'rfuleg', 'lfho', 'rfho'],
        'BLIMB': ['lblleg', 'lbuleg', 'rblleg', 'rbuleg', 'lbho', 'rbho'],
        # 'TAIL': ['tail']
    },

    'sheep':{
        'HEAD': ['head', 'lear', 'rear','leye', 'reye', 'muzzle', 'rhorn', 'lhorn'],
        'TORSO': ['torso', 'neck'],
        'FLIMB': ['lflleg', 'lfuleg', 'rflleg', 'rfuleg'],
        'BLIMB': ['lblleg', 'lbuleg', 'rblleg', 'rbuleg'],
        # 'TAIL': ['tail']
    }
}

part_merge_dict = {
    obj_class: { part:Part  for Part, parts in _part_cat_dir.items() for part in parts }
    for obj_class, _part_cat_dir in Part_dict.items()
}




# %%

here = os.path.dirname(os.path.realpath(__file__)) # get absoltae path to the dir this file is in
dataset_root=here+'/../__data__'
processed_root=dataset_root+'/processed'
metadata_dir = os.path.join(processed_root, 'metadata')
os.makedirs(metadata_dir, exist_ok=True)

from random import seed, sample
img_names = {}
with open(dataset_root+'/raw/VOC2010/ImageSets/Main/trainval.txt', 'r') as f:
    all_img_names = list(filter(lambda x:x!='', f.read().split('\n')))

seed(0)
img_names['train'] = list(sorted( sample(all_img_names, int(len(all_img_names) * 0.7) ) ))
img_names['val'] = list(sorted( list( set(all_img_names).difference(set(img_names['train'])) ) ))

for stage in ['train', 'val']:
    with open(os.path.join(metadata_dir, stage+'.txt'),'w') as f:
        for img_name in img_names[stage]:
            print(img_name, file=f)


args = argparse.Namespace()
args.annotation_folder = dataset_root+'/raw/metadata/Annotations_Part' # g3
args.images_folder = dataset_root+'/raw/VOC2010/JPEGImages' # g3

processed_dir=processed_root+'/%dx%d' %(size,size)
os.makedirs(processed_dir+'/obj_img/', exist_ok=True)
os.makedirs(processed_dir+'/obj_mask/', exist_ok=True)
os.makedirs(processed_dir+'/part_mask/', exist_ok=True)


obj_class_list = list(part_merge_dict.keys())
obj_num_dir = {obj_class:0 for obj_class in obj_class_list}
parts_dir = {obj_class:[] for obj_class in obj_class_list}

# Stats on the dataset:
obj_cnt = 0
bodypart_cnt = 0

mat_filenames = os.listdir(args.annotation_folder)

img_cnt_total = len(mat_filenames)
# Iterate through the .mat files contained in path:
for img_id, annotation_filename in enumerate(mat_filenames):
    annotations = utils.load_annotations(os.path.join(args.annotation_folder, annotation_filename))
    image_id = annotation_filename[:annotation_filename.rfind(".")]

    image_filename = image_id + ".jpg" # PASCAL VOC image have .jpg format
    img = PIL.Image.open(os.path.join(args.images_folder, image_filename))
    img = np.asarray( img, dtype=np.uint8 )

    for obj in annotations["objects"]:
        obj_class = obj["class"]
        obj_mask  = obj["mask"]
        if obj_class not in obj_class_list:
            continue

        img_crop, obj_mask_crop = crop(img, obj_mask)
        obj_cnt += 1
        obj_num_dir[obj_class] += 1
        obj_id = obj_class+'-' +str(obj_cnt-1) +'-'+image_id

        Part_mask_dict = {}

        for body_part in obj["parts"]:
            part_name = body_part["part_name"]
            part_mask = body_part["mask"]

            _, _, part_mask_crop = crop(img, obj_mask, part_mask)

            if part_name not in part_merge_dict[obj_class]:
                continue

            Part = part_merge_dict[obj_class][part_name]
            if Part not in Part_mask_dict.keys():
                Part_mask_dict[Part] = part_mask_crop
            else:
                Part_mask_dict[Part] = np.maximum(Part_mask_dict[Part],part_mask_crop)

            print("scanning raw: imgs {} / {} | already processed: obj_cnt {},  bodypart_cnt {}".
                format(img_id+1,img_cnt_total, obj_cnt, bodypart_cnt), end='\r')


        bodypart_cnt += len(Part_mask_dict) # len(obj["parts"])

        np_save_resize_img(img_crop, processed_dir+'/obj_img/'+obj_id)
        np_save_resize_img(obj_mask_crop, processed_dir+'/obj_mask/' +obj_id+'.mask')
        os.makedirs(processed_dir+'/part_mask/'+obj_id, exist_ok=True)
        for Part, Part_mask_crop  in Part_mask_dict.items():
            np_save_resize_img(Part_mask_crop, processed_dir+'/part_mask/'+obj_id+'/'+Part)

print('processed objects: ', obj_num_dir)
print("scanned raw: imgs {} / {} | processed: obj_cnt {},  bodypart_cnt {}".format(img_id+1,img_cnt_total, obj_cnt, bodypart_cnt))

# %%
