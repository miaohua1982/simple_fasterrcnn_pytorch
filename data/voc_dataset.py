from __future__ import  absolute_import

import os
import numpy as np
from config.config import running_args
from data.util import read_image, preprocess, resize_bbox, random_flip, flip_bbox

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class Voc_Dataset:
    def __init__(self, dataset_path, min_size, max_size, split='trainval', use_difficult=False):
        assert split in ['trainval','train','test','test']

        train_ds_cnt_path = os.path.join(dataset_path, 'ImageSets', 'Main', split+'.txt')
        f = open(train_ds_cnt_path)
        self.ids = [s.strip() for s in f.readlines()]
        f.close()

        self.dataset_base_path = dataset_path
        self.min_size = min_size
        self.max_size = max_size
        self.use_difficult = use_difficult

    def get_ds_labels(self):
        return VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def parse_one_annotation(self, idx):
        assert idx >= 0 and idx < len(self.ids)

        annotation_file_path = os.path.join(self.dataset_base_path, 'Annotations', self.ids[idx]+'.xml')
        from xml.dom.minidom import parse
        
        anno = parse(annotation_file_path)
        collection = anno.documentElement
        img_size = collection.getElementsByTagName('size')

        # image size
        w = img_size[0].getElementsByTagName('width')[0]
        width = w.childNodes[0].data
        h = img_size[0].getElementsByTagName('height')[0]
        height = h.childNodes[0].data
        # d = img_size[0].getElementsByTagName('depth')[0]
        # depth = d.childNodes[0].data

        # gt box & label
        # gt boxes
        gt_boxes = []
        gt_labels = []
        gt_difficults = []
        objects = collection.getElementsByTagName('object')
        for one_object in objects:
            name = one_object.getElementsByTagName('name')[0]
            obj_name = name.childNodes[0].data
            obj_name_idx = VOC_BBOX_LABEL_NAMES.index(obj_name)

            difficult = one_object.getElementsByTagName('difficult')[0]
            difficult = difficult.childNodes[0].data
            difficult = int(difficult)

            if not self.use_difficult and difficult == 1:
                continue
            
            bndbox = one_object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin = int(xmin.childNodes[0].data)-1
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin = int(ymin.childNodes[0].data)-1
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax = int(xmax.childNodes[0].data)-1
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax = int(ymax.childNodes[0].data)-1

            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_labels.append(obj_name_idx)
            gt_difficults.append(difficult)

        # read image
        image_file_path = os.path.join(self.dataset_base_path, 'JPEGImages', self.ids[idx]+'.jpg')
        img = read_image(image_file_path)

        img_size = (int(height), int(width))
        gt_boxes = np.stack(gt_boxes).astype(np.float32)
        gt_labels = np.array(gt_labels).astype(np.int32)        
        gt_difficults = np.array(gt_difficults).astype(np.uint8)        

        return img, img_size, gt_boxes, gt_labels, gt_difficults
        
    def __getitem__(self, idx):
        img, img_size, gt_boxes, gt_labels, gt_difficults = self.parse_one_annotation(idx)
        img = preprocess(img, running_args.min_img_size, running_args.max_img_size)
        gt_boxes = resize_bbox(gt_boxes, img_size, (img.shape[1], img.shape[2]))
        scale = img.shape[1]/img_size[0]

        # do img random flip to augment the dataset
        img, flip_param = random_flip(img, x_random=True, return_param=True)
        gt_boxes = flip_bbox(gt_boxes, (img.shape[1], img.shape[2]), x_flip=flip_param['x_flip'])
        
        return img, gt_boxes, gt_labels, gt_difficults, scale
