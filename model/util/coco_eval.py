import copy
import torch
import re
import sys
import numpy as np

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        values = [int(v) / 10 for v in values]
        result = {"bbox AP": values[0], "mask AP": values[12]}
        return result

class CocoEvaluator:
    def __init__(self, coco_gt, ann_labels, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        self.coco_results = []
    
    def accumulate(self): # input all predictions
        image_ids = list(set([res["image_id"] for res in self.coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_dt = self.coco_gt.loadRes(self.coco_results) if self.coco_results else COCO() # use the method loadRes

            coco_eval.cocoDt = coco_dt 
            coco_eval.params.imgIds = image_ids # ids of images to be evaluated
            coco_eval.evaluate() # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate() # 3s
    
    def summarize(self):
        temp = sys.stdout
        sys.stdout = TextArea()

        for iou_type in self.iou_types:
            print("IoU metric: {}".format(iou_type))
            self.coco_eval[iou_type].summarize()

        self.output = sys.stdout
        sys.stdout = temp

        return self.output
    
    '''
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))
            
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # convert to coco bbox format: xmin, ymin, w, h
            boxes = prediction["boxes"]
            x1, y1, x2, y2 = boxes.unbind(1)
            boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
            
            boxes = boxes.tolist()
            
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            coco_results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results
    
    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            rles = [
                mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results
    '''
    
    def prepare_for_one_coco(self, img_id, boxes, scores, labels, masks):
        if boxes.shape[0] == 0:
            return

        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1) # change back to (xmin,ymin,w,h)
        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        labels = [self.ann_labels[l] for l in labels]

        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        self.coco_results.extend(
            [
                {
                    "image_id": img_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )

    def finish_eval(self, result_path):
        if result_path is not None:
            torch.save(self.coco_results, result_path)
