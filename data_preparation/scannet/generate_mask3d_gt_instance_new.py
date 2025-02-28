import os
import yaml
import numpy as np
import torch
from pytorch3d.ops import knn_points
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('..')

from backproject import backprojector
# from mask2former_video.utils.util_3d import plot_3d

#  import ipdb
#  st = ipdb.set_trace

split = ['train', 'validation']
# split = ['two_scene']
# split = ['validation']
# split = ['train']

MASK3D_processed = '/mnt/data/odin_processed'
FRAME_DIR = '/mnt/data/odin_processed/frames_square_highres/'    

data = {}
total = 0
for s in split:
    with open(os.path.join(MASK3D_processed, f'{s}_database.yaml')) as f:
        data_ = yaml.load(f, Loader=yaml.FullLoader)
        # data_ is a list of dicts, with a key 'filepath'
        # eg: '/projects/katefgroup/language_grounding/mask3d_processed/scannet/train/0000_00.npy'
        # we need to get 0000_00 and use it as the key in data
        for d in data_:
            key = d['filepath'].split('/')[-1].split('.')[0]
            data[key] = d
            total += 1

print(f'Loaded {total} scenes')


def get_2d_labels(rgb_coords, mask3d_coords, labels, segments, depths, poses):
    N, H, W, _ = rgb_coords.shape
    
    # unproject our pc
    rgb_coords = backprojector([rgb_coords.permute(0, 3, 1, 2)], depths[None].float(), poses[None].float())[0][0]
    rgb_coords = rgb_coords.reshape(1, -1, 3)
    
    # do 1 NN with mask3d points
    dists_idxs = knn_points(rgb_coords, mask3d_coords[None])
    mask3d_idxs = dists_idxs.idx.squeeze(-1).cpu().numpy()
    
    # valids
    valids = (dists_idxs.dists.squeeze(-1) < 0.1) & (depths != 0).flatten()
    valids = valids.reshape(N, H, W)
    
    # labels
    rgb_coord_labels = labels[mask3d_idxs[0]]
    rgb_coord_labels = rgb_coord_labels[:, 0] * 1000 + rgb_coord_labels[:, 1] + 1
    rgb_coord_labels = rgb_coord_labels.reshape(N, H, W)
    
    # segments
    rgb_coord_segments = segments[mask3d_idxs]
    rgb_coord_segments = rgb_coord_segments.reshape(N, H, W)
    
    # return per image labels, segments and valids
    return rgb_coord_labels, rgb_coord_segments.cpu().numpy(), valids.cpu().numpy()


def save(data, folder_name, type_, scene, fname):
    im = Image.fromarray(data.astype(type_))
    if not os.path.exists(os.path.join(FRAME_DIR, scene, folder_name)):
        os.mkdir(os.path.join(FRAME_DIR, scene, folder_name))
    im.save(os.path.join(FRAME_DIR, scene, folder_name, fname))



count = 0
for scene in tqdm(os.listdir(FRAME_DIR)):
    scene_id = scene.split('scene')[1]
    
    if scene_id not in data:
        continue
    
    # load labels from data
    points = np.load(data[scene_id]['filepath'], allow_pickle=True)
    coordinates, mask3d_colors, _, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )
    
    coordinates = torch.from_numpy(coordinates).cuda()
    segments = torch.from_numpy(segments).cuda()
    
    if labels.shape[0] != coordinates.shape[0]:
        print(f'{labels.shape[0]} != {coordinates.shape[0]} for {scene_id}')
        print(scene_id)
        continue
    
    fnames = sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))
    
    images = np.stack(
            [np.array(Image.open(os.path.join(FRAME_DIR, scene, 'color', frame))) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))])
    images = torch.from_numpy(images).cuda().float()
    
    depths = np.stack(
        [np.array(Image.open(os.path.join(FRAME_DIR, scene, 'depth_inpainted', frame))) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'depth')))])
    depths = torch.from_numpy(depths.astype(np.float32)).cuda().float() / 1000.0
    
    poses = np.stack(
            [np.loadtxt(os.path.join(FRAME_DIR, scene, 'pose', frame)) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'pose')))]
        )
    poses = torch.from_numpy(poses).cuda().float()
    
    labels_2d, valids_2d, mask_segments_2d = get_2d_labels(images, coordinates, labels, segments, depths, poses)
    
    for i, label in enumerate(labels_2d):
        # store label as png image
        save(label, 'instance_sep', np.int32, scene, fnames[i].replace('jpg', 'png'))
        save(valids_2d[i], 'valids', bool, scene, fnames[i].replace('jpg', 'png'))
        save(mask_segments_2d[i], 'segments', np.int32, scene, fnames[i].replace('jpg', 'png'))
        
    