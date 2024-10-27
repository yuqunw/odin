# Import torch Dataset
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from PIL import Image
import torch
import numpy as np
import argparse
from odin.data_video.scannet_3d_eval import Scannet3DEvaluator
from tqdm import tqdm
from odin.data_video.datasets.scannet_context import LEARNING_MAP20
import torch.nn.functional as F
# Import voxelization
from odin.utils import voxelization
from odin.modeling.backproject.backproject import \
    multiscsale_voxelize, interpolate_feats_3d, voxelization, voxel_map_to_source, \

LEARNING_MAP20 = {0: 0, 1: 19, 2: 20, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6,  9: 7,
                10: 8, 11: 9, 12: 10, 13: 0, 14: 11, 15: 0, 16: 12, 17: 0,
                18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 13, 25: 0, 26: 0,
                27: 0, 28: 14, 29: 0, 30: 0, 31: 0, 32: 0, 33: 15, 34: 16, 
                35: 0, 36: 17, 37: 0, 38: 0, 39: 18, 40: 0, 50: 0, 65: 0}

def get_valid_points(depth, pose, img_coor, intrinsic):
    # K = torch.tensor([577.590698, 0.000000, 318.905426,
    #                   0.000000, 578.729797, 242.683609,
    #                   0.000000, 0.000000, 1.000000,]).reshape(3,3)
    K = intrinsic[:3, :3] # 3x3
    h, w = depth.shape[-2:]
    cam_coor = img_coor @ torch.inverse(K).T * depth.permute(1, 2, 0) #h, w, 3 @ 3, 3 -> h, w, 3 * h, w, 1 -> h, w, 3

    ones = torch.ones_like(cam_coor[..., :1]) # h, w, 1
    h_points = torch.cat((cam_coor, ones), -1).view(-1, 4) # h, w, 4 -> h*w, 4

    # transform using pose to world coordinate
    projected_mask_cam_coor = (pose @ h_points.t().float()).t()[..., :3].view(h, w, 3)

    # projected_mask_cam_coor = mask_cam_coor.float() @ pose[:3, :3].T - pose[:3, 3]
    return projected_mask_cam_coor

def get_img_coor(h=480, w=640):
    y, x = torch.from_numpy(np.mgrid[:h, :w])
    img_coor = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return img_coor   

class eval_dataset(Dataset):
    # Write the __init__ method
    def __init__(self, data_dir='/mnt/data/odin_processed', 
                       mask_dir='/home/yuqun/research/multi_purpose_nerf/Tracking-Anything-with-DEVA/eval_dir/odin_gt/', 
                       score_dir='/home/yuqun/research/feed_forward/odin/outputs/gt_depth_rgbd',
                       depth_prefix='depth_inpainted'):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.score_dir = score_dir
        self.depth_prefix = depth_prefix
        self.depth_files, self.mask_files, self.score_files, self.pc_files, self.pose_files, self.intrinsic_files = self.get_files()
    
    # Write the __len__ method
    def __len__(self):
        return len(self.depth_files)

    def get_files(self):
        validate_scene_list = sorted([i.name for i in Path('/mnt/data/scannet_region_usage/val').iterdir()])
        depth_files = []
        mask_files = []
        score_files = []
        pc_files = []
        pose_files = []
        intrinsic_files = []

        for scene_name in validate_scene_list:
            depth_dir = Path(self.data_dir) / 'frames_square_highres' / scene_name / self.depth_prefix
            pose_dir = Path(self.data_dir) / 'frames_square_highres' / scene_name / 'pose'
            mask_dir = Path(self.mask_dir) / scene_name
            score_dir = Path(self.score_dir) / scene_name

            pc_file = Path(self.data_dir) / 'validation' / f'{scene_name}.npy'
            intrinsic_files = Path(self.data_dir) / 'frames_square_highres' / scene_name / 'intrinsic_depth.txt'

            scene_depth_files = sorted(list(depth_dir.iterdir()))
            scene_mask_files = sorted(list(mask_dir.iterdir()))
            scene_score_files = sorted(list(score_dir.iterdir()))
            scene_pose_files = sorted(list(pose_dir.iterdir()))

            depth_files.extend(scene_depth_files)
            mask_files.extend(scene_mask_files)
            score_files.extend(scene_score_files)
            pose_files.extend(scene_pose_files)
            pc_files.extend(pc_file)
            intrinsic_files.append(intrinsic_files)
        
        return depth_files, mask_files, score_files, pc_files, pose_files, intrinsic_files

    # Write the __getitem__ method
    def __getitem__(self, idx):
        scene_depth_files = self.depth_files[idx]
        scene_mask_files = self.mask_files[idx]
        scene_score_files = self.score_files[idx]
        scene_pc_file = self.pc_files[idx]
        scene_pose_file = self.pose_files[idx]
        scene_intrinsic_file = self.intrinsic_files[idx]

        scene_depths = []
        scene_masks = []
        scene_scores = []
        scene_labels = []
        scene_poses = []

        for depth_file, mask_file, score_file, pose_file in zip(scene_depth_files, scene_mask_files, scene_score_files, scene_pose_file):
            pose = np.loadtxt(pose_file)

            if (pose == -np.inf).any():
                continue
                
            scene_depth = torch.tensor(cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 1000.)
            scene_mask = torch.tensor(cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED))
            scene_score = torch.load(score_file)
            scene_pose = torch.tensor(pose).float()

            scene_depths.append(scene_depth)
            scene_masks.append(scene_mask)
            scene_scores.append(torch.tensor(scene_score['scores']))
            scene_labels.append(torch.tensor(scene_score['pred_classes']))
            scene_poses.append(scene_pose)

        scene_depths = torch.stack(scene_depths, dim=0)
        scene_masks = torch.stack(scene_masks, dim=0)
        scene_scores = torch.stack(scene_scores, dim=0)
        scene_labels = torch.stack(scene_labels, dim=0)
        scene_poses = torch.stack(scene_poses, dim=0)
        

        # 
        
        sample = {'depths': scene_depths, 
                  'pred_masks': scene_masks, 
                  'scores': scene_scores, 
                  'pred_labels': scene_labels,
                  'poses': scene_poses}

        K = np.loadtxt(scene_intrinsic_file)
        K = torch.tensor(K).float()[:3, :3]
        sample['K'] = K

        scene_pc = np.load(scene_pc_file)
        segments = scene_pc[:, 9]
        sample['segments'] = torch.from_numpy(segments).long()
        labels = torch.from_numpy(scene_pc[:, 10:12])

        unique_instances = torch.unique(labels[:, 1])
        if len(unique_instances) > 0 and unique_instances[0] == -1:
            unique_instances = unique_instances[1:]
        
        num_unique_instances = len(unique_instances)
        scannet_masks = []
        scannet_classes = []

        for k in range(num_unique_instances):
            scannet_mask = labels[:, 1] == unique_instances[k]
            class_label = labels[:, 0][scannet_mask][0]
            if class_label.item() not in LEARNING_MAP20:
                # print(f"{class_label.item()} not in learning map")
                continue
            class_label = LEARNING_MAP20[class_label.item()]
            if class_label == 0:
                continue

            scannet_masks.append(scannet_mask)
            scannet_classes.append(class_label - 1)

        scannet_masks = torch.stack(scannet_masks, dim=0)
        scannet_classes = torch.tensor(scannet_classes, dtype=torch.int64)

        sample['scannet_masks'] = scannet_masks
        sample['scannet_classes'] = scannet_classes
        

        # Coordinate
        scene_coords = torch.from_numpy(scene_pc[:, :3])
        sample['coords'] = scene_coords
        
        return sample
    

def evaluate_pc_instance(args, eval_dataset, down_scale=True):
    evaluator = Scannet3DEvaluator(dataset_name='scannet', output_dir=f'{args.save_path}/evaluation')
    evaluator.reset()

    scene_voxel_size=0.02
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if down_scale:
        H, W = 240, 320

    img_coors = get_img_coor(H, W).float().to(device)

    for idx, sample in enumerate(tqdm(eval_dataset)):
        depth_images = sample['depths'].to(device)
        pred_masks = sample['pred_masks'].to(device)
        pred_scores = sample['scores'].to(device)
        pred_labels = sample['pred_labels'].to(device)
        poses = sample['poses'].to(device)
        intrinsics = sample['K'].to(device)
        segments = sample['segments'].to(device)
        scannet_masks = sample['scannet_masks'].to(device)
        scannet_classes = sample['scannet_classes'].to(device)
        scene_points = sample['coords'].to(device)

        image_pc_instance_list = []
        image_pc_xyz_list = []

        if down_scale:
            pred_masks = F.interpolate(pred_masks[None], scale_factor=0.5, mode='nearest', )[0]
            depth_images = F.interpolate(depth_images[None,], scale_factor=0.5, mode='nearest', )[0]
            intrinsics[:2, :] *= 0.5
        
        for file_index, pose in enumerate(poses):
            depth_image = depth_images[file_index]
            pred_mask = pred_masks[file_index]
            point = get_valid_points(depth_image, pose, img_coors, intrinsics) # H, W, 3   
            valid_mask = (depth_image[0] > 0)
            point = point[valid_mask]

            image_pc_xyz_list.append(point) # [num_points, 3]
            # image_pc_semantic_list.append(semantic_pixel_prediction.view(H*W, -1)[valid_mask.flatten(0,1)]) # [num_points, slot_num]
            image_pc_instance_list.append(pred_mask.view(H*W, -1).permute(1,0)[valid_mask.flatten(0,1)]) # [num_points, slot_num]            

        pc_xyz = torch.cat(image_pc_xyz_list, dim=0) # [num_points, 3]
        pc_instance = torch.cat(image_pc_instance_list, dim=0) # [num_points, slot_num]
        # pc_semantic = torch.cat(image_pc_semantic_list, dim=0) # [num_points, output_dim]
        pc_p2v = voxelization(pc_xyz[None], scene_voxel_size)[0] # [num_points, 3]

        # Clean memory
        torch.cuda.empty_cache()            

    return 


if __name__ == 'main':
    # Add arguments
    parser = argparse.ArgumentParser(description='Evaluate ODIN outputs')

    # Add arguments
    parser.add_argument('--data_dir', type=str, default='/mnt/data/odin_processed')
    parser.add_argument('--mask_dir', type=str, default='/home/yuqun/research/multi_purpose_nerf/Tracking-Anything-with-DEVA/eval_dir/odin_gt/')
    # save path
    parser.add_argument('--save_path', type=str, default='/outputs/gt_depth_rgbd')

    # Parse arguments   
    args = parser.parse_args()

    eval_dataset = eval_dataset()

    evaluate_pc_instance(args, eval_dataset)
