from pathlib import Path
import os
import shutil

source_dir = Path('/mnt/data/scannet_region_usage/val')
target_dir = Path('/mnt/data/odin_processed/frames_square_highres')

for scene_dir in sorted(source_dir.iterdir()):
    scene_name = scene_dir.name 
    mast3r_dir = scene_dir / 'depth_mast3r'
    target_scene_dir = target_dir / scene_name / 'depth_mast3r'

    if not target_scene_dir.exists():
        os.makedirs(target_scene_dir)

    for frame in mast3r_dir.iterdir():
        if not frame.is_file():
            continue
        print(f'{int(frame.stem)}.png')
        target_frame = target_scene_dir / f'{int(frame.stem)}.png'
        shutil.copy(frame, target_frame)
        print(f'Copied {frame} to {target_frame}')