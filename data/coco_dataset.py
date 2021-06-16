from __future__ import  absolute_import

import os
import numpy as np
import torch as t
from pycocotools.coco import COCO
from data.util import read_image, preprocess, resize_bbox, resize_mask, random_flip, flip_bbox, flip_mask, xmin_ymin_wh_2_xyxy

class Coco_Dataset(object):
    def __init__(self, basedir, min_size, max_size, split):
        ann_file_path = os.path.join(basedir, 'annotations/instances_{}.json'.format(split))
        self.basedir = basedir
        self.split = split
        self.img_min_size = min_size
        self.img_max_size = max_size
        self.coco = COCO(ann_file_path)
        self.img_ids = [id for id in self.coco.imgs]
        self.classes_map = {k:v['name'] for k, v in self.coco.cats.items()}
        self.classes_id_map = {i:k for i, k in enumerate(sorted(self.classes_map))}
        #self.classes_to_idx = {self.classes_map[k]:i for i, k in enumerate(sorted(self.classes_map))}
        self.classes_labels = [self.classes_map[k] for k in sorted(self.classes_map)]

        self.num_classes = len(self.classes_map)

    def get_ds_labels(self):
        return self.classes_labels

    def __len__(self):
        return len(self.img_ids)

    def get_class_num(self):
        return self.num_classes
    
    def get_class_name(self, label):
        return self.classes_labels[label]

    def __getitem__(self, idx):
        # image id
        img_id = self.img_ids[idx]
        # np.ndarray ~ [C,H,W]
        img = self.load_img(img_id)
        prop = self.get_other_prop(img_id)

        # np.ndarray ~ resize to config size, [C,H,W]
        pre_img_size = (img.shape[1], img.shape[2])
        img = preprocess(img, min_size=self.img_min_size, max_size=self.img_max_size)
        post_img_size = (img.shape[1], img.shape[2])

        # scale
        prop['scale'] = post_img_size[0]/pre_img_size[0]

        # if there exists gt boxes, do resize & flip
        if len(prop['gt_boxes']) > 0 :
            # resize boxes & masks
            prop['gt_boxes'] = resize_bbox(prop['gt_boxes'], pre_img_size, post_img_size)
            prop['gt_masks'] = resize_mask(prop['gt_masks'], pre_img_size, post_img_size)

            # do img random flip to augment the dataset
            img, flip_param = random_flip(img, x_random=True, return_param=True)
            prop['gt_boxes'] = flip_bbox(prop['gt_boxes'], post_img_size, x_flip=flip_param['x_flip'])
            prop['gt_masks'] = flip_mask(prop['gt_masks'], x_flip=flip_param['x_flip'])

        # note mask's dtype is uint8
        return img, prop 

    def load_img(self, img_id):
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = os.path.join(self.basedir, self.split, img_name)
        img = read_image(img_path)

        return img

    def get_other_prop(self, img_id):
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        is_crowd = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                is_crowd.append(ann['iscrowd'])
                name = self.classes_map[ann["category_id"]]
                labels.append(self.classes_labels.index(name)) # label is from 0~79
                mask = self.coco.annToMask(ann)
                mask = np.array(mask, dtype=np.uint8)
                masks.append(mask)

            boxes = xmin_ymin_wh_2_xyxy(np.array(boxes, dtype=np.float32))
            labels = np.array(labels)
            is_crowd = np.array(is_crowd)
            masks = np.stack(masks)
            # you can use following code to generate boxes from mask
            # the only different from coco api annToMask is, my code generates interger axises
            # and api generates float versions, there are maybe a little differences in value
            # boxes = self.getbox_from_mask(masks.numpy())
        else:  # well, there may be exceptions, no annotation in a picture
            boxes = np.array(boxes)
            labels = np.array(boxes)
            masks = np.array(masks)
            is_crowd = np.array(is_crowd)
        
        prop = dict(image_id=img_id, iscrowd=is_crowd, gt_boxes=boxes, gt_labels=labels, gt_masks=masks)

        return prop
    
    def getbox_from_mask(self, gt_masks):
        boxes = []
        num = gt_masks.shape[0]
        for i in range(num):
            one_mask = gt_masks[i]
            h, w = one_mask.shape
            xs = np.arange(w)[np.any(one_mask==1, axis=0)]
            if xs.shape[0] == 0 :
                continue
            xmin, xmax = xs[0], xs[-1]
            
            ys = np.arange(h)[np.any(one_mask==1, axis=1)]
            if ys.shape[0] == 0:
                continue
            ymin, ymax = ys[0], ys[-1]
            
            boxes.append([xmin, ymin, xmax, ymax])

        return np.array(boxes)

if __name__ == '__main__':
    coco = Coco_Dataset('E:\\Datasets\\COCO2017',800, 1024, 'train2017')
    img, prop = coco[0]
    print(prop['scale'])