from pathlib import Path
import os
import shutil

source_dir = Path('/mnt/data/scannet_region_usage/val')

for scene_dir in sorted(source_dir.iterdir()):
    scene_name = scene_dir.name 
    mast3r_dir = scene_dir / 'depth_mast3r'
    if mast3r_dir.exists():
        shutil.rmtree(mast3r_dir)

    mast3r_pos_dir = scene_dir / 'pos_encoding' / 'scale_and_rotated_pth_mast3r/'

    if mast3r_pos_dir.exists():
        shutil.rmtree(mast3r_pos_dir)