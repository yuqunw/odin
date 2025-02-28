set -e

export DETECTRON2_DATASETS="/mnt/data/odin_processed"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train_odin.py  --dist-url='tcp://127.0.0.1:8474' --num-gpus 1  --resume --config-file configs/scannet_context/swin_3d.yaml \
OUTPUT_DIR output/test_swin \
SOLVER.IMS_PER_BATCH 1 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 9 INPUT.FRAME_RIGHT 9 INPUT.SAMPLING_FRAME_NUM 19 \
MODEL.WEIGHTS 'https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco_swin.pkl' \
SOLVER.BASE_LR 1e-4 \
INPUT.IMAGE_SIZE 256 \
MODEL.CROSS_VIEW_CONTEXTUALIZE True \
INPUT.CAMERA_DROP True \
INPUT.STRONG_AUGS True \
INPUT.AUGMENT_3D True \
INPUT.VOXELIZE True \
INPUT.SAMPLE_CHUNK_AUG True \
MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
MODEL.CROSS_VIEW_BACKBONE True \
DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k_rgbd',)" \
DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k_rgbd','scannet_context_instance_train_eval_20cls_single_highres_100k_rgbd',)" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES "[19, 20]" \
USE_GHOST_POINTS False \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
USE_SEGMENTS False \
SOLVER.MAX_ITER 100000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM -1 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
MODEL.SUPERVISE_SPARSE True \
HIGH_RES_INPUT True \
# MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/m2f_coco.pkl' \
# reduce lr at 64k iterations to 1e-5, get the best instance segmentation checkpoint \
# at 6k iterations and best semantic segmentation checkpoint at 2k iterations
