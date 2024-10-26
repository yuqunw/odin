
# SAVING DIRECTORY - CHANGE THIS
DATA_DIR = '/mnt/data/odin_processed/ai2thor/SEMSEG_100k_reproduce/ai2thor_frames_512'
SPLITS_PATH = 'splits/ai2thor_splits'

SPLITS = {
    'train':   f'{SPLITS_PATH}/ai2thor_train.txt',
    'val':     f'{SPLITS_PATH}/ai2thor_val.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
