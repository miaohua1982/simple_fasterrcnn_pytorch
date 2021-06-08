import numpy as np
import os
import skimage
import skimage.io
from matplotlib import pyplot as  plt
from pycocotools.coco import COCO

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


if __name__ == '__main__':
    ann_path = 'E:\\Datasets\\COCO2017\\annotations\\instances_train2017.json'
    split = 'train2017'
    show_one_img_with_ann_by_path(ann_path, split)
