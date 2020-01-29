#!/usr/bin/env python
# coding: utf-8
"""Given the dataset as the image folders as classnames, create the lmdb for caffe pre-processing"""

import os
import shutil
from tqdm import tqdm

train_set_path = '/media/achyut/storage/sorted_images/consolidated_dataset_for_keras_consumption/trainset_for_keras'
test_set_path = '/media/achyut/storage/sorted_images/consolidated_dataset_for_keras_consumption/testset_for_keras'
target_class_names = ['empty', 'multi-old', 'single']
label_map = dict(zip(target_class_names, range(len(target_class_names))))


train_txt_file = '/home/achyut/Desktop/classifier_sorted_rgb/train.txt'
test_txt_file = '/home/achyut/Desktop/classifier_sorted_rgb/test.txt'
consolidated_images_folder = '/home/achyut/Desktop/classifier_sorted_rgb/all_rgb_images'


if not os.path.exists(consolidated_images_folder):
    os.makedirs(consolidated_images_folder)

def consolidate_images(set_path, label_map, consolidated_images_folder, txt_file):
    lines = []
    for cls in tqdm(os.listdir(set_path)):
        cls_path = os.path.join(set_path, cls)
        for image_name in os.listdir(cls_path):
            image_path = os.path.join(cls_path, image_name)
            dest_image_path = os.path.join(consolidated_images_folder, image_name)
            if not os.path.exists(dest_image_path):
                shutil.copyfile(image_path, dest_image_path)
            lines.append(image_name + ' ' + str(label_map[cls]) + '\n')
    with open(txt_file, 'w') as fp:
        fp.writelines(lines)        

consolidate_images(train_set_path, label_map, consolidated_images_folder, train_txt_file)
consolidate_images(test_set_path, label_map, consolidated_images_folder, test_txt_file)


lmdb_path = '/home/achyut/Desktop/classifier_sorted_rgb/lmdb'
train_lmdb_path = lmdb_path+'/train_lmdb'
test_lmdb_path = lmdb_path+'/test_lmdb'
if not os.path.exists(lmdb_path):
    os.makedirs(lmdb_path)

# Run the following commands in the terminal if you don't have ipython

get_ipython().system(u"GLOG_logtostderr=1 '/home/achyut/projects/caffe/build/tools/convert_imageset'     --resize_height=224 --resize_width=224 --shuffle      {consolidated_images_folder+'/'}     {train_txt_file}     {train_lmdb_path}")

get_ipython().system(u"GLOG_logtostderr=1 '/home/achyut/projects/caffe/build/tools/convert_imageset'     --resize_height=224 --resize_width=224 --shuffle      {consolidated_images_folder+'/'}     {test_txt_file}     {test_lmdb_path}")

get_ipython().system(u"GLOG_logtostderr=1 '/home/achyut/projects/caffe/build/tools/compute_image_mean'     {train_lmdb_path}     {lmdb_path+'/'+'mean.binaryproto'}")


