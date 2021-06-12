import numpy as np
import os
import skimage
import skimage.io
from skimage import measure
from matplotlib import pyplot as  plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def show_one_img_by_path(img_path):
    img = skimage.io.imread(img_path)
    plt.imshow(img)
    plt.show()


def show_one_img_with_ann_by_path(ann_path, split):
    coco = COCO(ann_path)
    catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard']) # [1, 18, 41]
    imgIds = coco.getImgIds(catIds=catIds)
    # randomly choose one
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    print('we randomly choose following img')
    print(img)
    # img path
    img_path = os.path.join(os.path.dirname(ann_path), '../{}/{}'.format(split, img['file_name']))
    plt.subplot(121)
    plt.title('Original_Img')
    show_one_img_by_path(img_path)

    # show img by anns
    plt.subplot(122)
    plt.title('Ann_Img')
    show_one_img_by_path(img_path)
    # show anns
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)     
    anns = coco.loadAnns(annIds)                                           
    coco.showAnns(anns)
    # coco for person key points
    pkp_ann_path = os.path.join(os.path.dirname(ann_path), 'person_keypoints_{}.json'.format(split))
    coco_kps = COCO(pkp_ann_path)
    kps_annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)     
    kps_anns = coco_kps.loadAnns(kps_annIds)                                           
    coco_kps.showAnns(kps_anns)
    # coco for caption
    cap_ann_path = os.path.join(os.path.dirname(ann_path), 'captions_{}.json'.format(split))
    coco_caps = COCO(cap_ann_path)
    caps_annIds = coco_caps.getAnnIds(imgIds=img['id'])
    caps_anns = coco_caps.loadAnns(caps_annIds)
    coco_caps.showAnns(caps_anns)


def maskToanno(ground_truth_binary_mask, img_id, ann_id, category_id):
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = maskUtils.area(encoded_ground_truth)
    ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": 0,
        "image_id": img_id,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": category_id,
        "id": ann_id
    }
    for contour in contours:
        # find_contours returns consisting of n ``(row, column)`` coordinates along the contour
        # we want to change to (x,y), so flip it
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    return annotation

def vis_mask_to_pic(ann_path, split):
    coco = COCO(ann_path)
    catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard']) # [1, 18, 41]
    imgIds = coco.getImgIds(catIds=catIds)
    # randomly choose one
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    img_id = img['id']
    print('we randomly choose following img')
    print(img)
    # img path
    img_path = os.path.join(os.path.dirname(ann_path), '../{}/{}'.format(split, img['file_name']))
    plt.subplot(131)
    plt.title('Original_Img')
    show_one_img_by_path(img_path)
    
    # show my anns
    annIds = coco.getAnnIds(img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    my_anns = []
    for idx, ann in enumerate(anns):
        cat_id = ann["category_id"]
        gt_mask = coco.annToMask(ann)
        print(idx, gt_mask.shape)
        gt_mask = np.array(gt_mask, dtype=np.uint8)
        one_ann = maskToanno(gt_mask, img_id, idx, cat_id)
        my_anns.append(one_ann)

    # show img by my anns
    plt.subplot(132)
    plt.title('MyAnn_Img')
    show_one_img_by_path(img_path)
    coco.showAnns(my_anns)

    # show img by anns
    plt.subplot(133)
    plt.title('Ann_Img')
    show_one_img_by_path(img_path)
    # show anns
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)     
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

if __name__ == '__main__':
    ann_path = 'E:\\Datasets\\COCO2017\\annotations\\instances_train2017.json'
    split = 'train2017'
    vis_mask_to_pic(ann_path, split)
    show_one_img_with_ann_by_path(ann_path, split)
