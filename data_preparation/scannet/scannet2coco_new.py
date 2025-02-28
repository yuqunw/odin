import os
import argparse
import datetime
import csv
import torch
import numpy as np
import json

from PIL import Image
from random import random
from pycococreatortools import pycococreatortools
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import pycocotools.mask as mask_util
import pycocotools
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from data_preparation.scannet.globals_dirs import DATA_DIR, SPLITS

#  import ipdb
#  st = ipdb.set_trace

LEARNING_MAP = {0: 0, 1: 19, 2: 20, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6,  9: 7,
                10: 8, 11: 9, 12: 10, 13: 0, 14: 11, 15: 0, 16: 12, 17: 0,
                18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 13, 25: 0, 26: 0,
                27: 0, 28: 14, 29: 0, 30: 0, 31: 0, 32: 0, 33: 15, 34: 16, 
                35: 0, 36: 17, 37: 0, 38: 0, 39: 18, 40: 0, 50: 0, 65: 0}

NAME_MAP = { 1: 'cabinet', 2: 'bed', 3: 'chair', 4: 'sofa', 5: 'table', 6: 'door', 7: 'window',
             8: 'bookshelf', 9: 'picture', 10: 'counter', 11: 'desk', 12: 'curtain', 13: 'refridgerator',
             14: 'shower curtain', 15: 'toilet', 16: 'sink', 17:' bathtub',  18: 'otherfurniture',
             19: 'wall', 20: 'floor'}



# SPLITS = {
#     'train':   'scannet_splits/scannetv2_train.txt',
#     'val':     'scannet_splits/scannetv2_val.txt',
#     'train20': 'scannet_splits/data_efficient_by_images/scannetv2_train_20.txt',
#     'train40': 'scannet_splits/data_efficient_by_images/scannetv2_train_40.txt',
#     'train60': 'scannet_splits/data_efficient_by_images/scannetv2_train_60.txt',
#     'train80': 'scannet_splits/data_efficient_by_images/scannetv2_train_80.txt',
#     'val50': 'scannet_splits/scannetv2_val50.txt',
#     'val20': 'scannet_splits/scannetv2_val20.txt',
#     'two_scene': 'scannet_splits/two_scene.txt',
#     'one_scene': 'scannet_splits/one_scene.txt',
#     'debug': 'scannet_splits/scannetv2_debug.txt',
#     'ten_scene': 'scannet_splits/data_efficient_by_images/scannetv2_train_10_scene.txt'
# }

INFO = {
    "description": "ScanNet Dataset",
    "url": "https://github.com/sekunde",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Ji Hou",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP.items() 
]

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def polygons_to_bitmask(polygons, height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def convert_scannet_to_coco(path, phase):
    transform = Resize([240,320], Image.NEAREST)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "depths": [],
        "poses": [],
        "valids": [],
        "segments": [],
        "annotations": []
    }

    # get list
    scene_ids = read_txt(SPLITS[phase])
    image_ids = []
    for scene_id in scene_ids:
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    coco_ann_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]
        ann_path = os.path.join(path, scene_id, 'instance_sep', image_id + '.png')
        ann_map = Image.open(ann_path)
        ann_map = transform(ann_map)
        image_size = ann_map.size
        ann_map = np.array(ann_map)

        ann_ids = np.unique(ann_map)
        print(ann_path)
        for ann_id in ann_ids:
            label_id = LEARNING_MAP[int(ann_id / 1000)]
            inst_id = int(ann_id % 1000)
            if label_id == 0:
                continue

            # is_crowd helps in making pycoco run RLE instead of polygons
            category_info = {'id': label_id, 'is_crowd': 1}
            binary_mask = (ann_map == ann_id).astype(np.uint8)
            mask_size = binary_mask.sum()

            if mask_size == 0:
                continue

            #if mask_size < 1000:
            #    continue
            # st()
            ann_info = pycococreatortools.create_annotation_info(
                coco_ann_id, coco_image_id, category_info, binary_mask,
                image_size, tolerance=0)
            ann_info['iscrowd'] = 0
            
            rle = ann_info['segmentation']
            rle =pycocotools.mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
            rle['counts'] = rle['counts'].decode('ascii')
            ann_info['segmentation'] = rle
            
            ann_info['semantic_instance_id_scannet'] = label_id*1000+inst_id

            if ann_info is not None:
                coco_output['annotations'].append(ann_info)
                coco_ann_id += 1

        valid_filename = os.path.join(path, scene_id, 'valids', image_id + '.png')
        # if not os.path.exists(valid_filename):
        #     st()
            # print("valid file not exist: {}, creating one".format(valid_filename))
            # valid = np.ones((240,320))
            # np.save(valid_filename, valid)

        valid_info = pycococreatortools.create_image_info(coco_image_id, valid_filename, image_size)
        coco_output['valids'].append(valid_info)
        
        segment_filename = os.path.join(path, scene_id, 'segments', image_id + '.png')
        segment_info = pycococreatortools.create_image_info(coco_image_id, segment_filename, image_size)
        coco_output['segments'].append(segment_info)

        image_filename = os.path.join(scene_id, 'color', image_id + '.png')
        image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
        coco_output['images'].append(image_info)

        depth_filename = os.path.join(scene_id, 'depth', image_id + '.png')
        depth_info = pycococreatortools.create_image_info(coco_image_id, depth_filename, image_size)
        coco_output['depths'].append(depth_info)

        pose_filename = os.path.join(scene_id, 'pose', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(coco_image_id, pose_filename, image_size)
        coco_output['poses'].append(pose_info)
        coco_image_id += 1

    json.dump(coco_output, open(f'/mnt/data/odin_processed/scannet20_{phase}.coco.json','w'))


def config():
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', default='/mnt/data/odin_processed/frames_square_highres')
    parser.add_argument('--phase', default='train')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = config()
    phases = ['train', 'val', 'ten_scene', 'two_scene']
    # phases = ['two_scene']
    for phase in phases:
        convert_scannet_to_coco(opt.scannet_path, phase)