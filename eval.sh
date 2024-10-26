#!/bin/bash

# Set environment variables
export DETECTRON2_DATASETS="/mnt/data/odin_processed"
export OMP_NUM_THREADS="8"
export CUDA_VISIBLE_DEVICES="0"

# # Run the training script with arguments
# python train_odin.py \
#     --dist-url=tcp://127.0.0.1:8474 \
#     --num-gpus 1 \
#     --resume \
#     --config-file configs/scannet_context/swin_3d.yaml \
#     --eval-only \
#     OUTPUT_DIR outputs/mast3r_with_segment \
#     SOLVER.IMS_PER_BATCH 4 \
#     SOLVER.CHECKPOINT_PERIOD 4000 \
#     TEST.EVAL_PERIOD 4000 \
#     INPUT.FRAME_LEFT 9 \
#     INPUT.FRAME_RIGHT 9 \
#     INPUT.SAMPLING_FRAME_NUM 19 \
#     MODEL.WEIGHTS checkpoints/scannet_swin_50.0_64k_6k.pth \
#     SOLVER.BASE_LR 1e-4 \
#     INPUT.IMAGE_SIZE 256 \
#     MODEL.CROSS_VIEW_CONTEXTUALIZE True \
#     INPUT.CAMERA_DROP True \
#     INPUT.STRONG_AUGS True \
#     INPUT.AUGMENT_3D True \
#     INPUT.VOXELIZE True \
#     INPUT.SAMPLE_CHUNK_AUG True \
#     MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
#     MODEL.CROSS_VIEW_BACKBONE True \
#     DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k',)" \
#     DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k')" \
#     MODEL.PIXEL_DECODER_PANET True \
#     MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
#     MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
#     SKIP_CLASSES "[19, 20]" \
#     USE_GHOST_POINTS True \
#     MODEL.FREEZE_BACKBONE False \
#     SOLVER.TEST_IMS_PER_BATCH 1 \
#     SAMPLING_STRATEGY consecutive \
#     USE_SEGMENTS True \
#     SOLVER.MAX_ITER 100000 \
#     DATALOADER.NUM_WORKERS 8 \
#     DATALOADER.TEST_NUM_WORKERS 2 \
#     MAX_FRAME_NUM -1 \
#     MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
#     MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
#     USE_WANDB False \
#     USE_MLP_POSITIONAL_ENCODING True \
#     HIGH_RES_INPUT False \
#     EVAL_PER_IMAGE False \
#     DEPTH_PREFIX depth_mast3r \
#     MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
#     MODEL.MASK_FORMER.TEST.INSTANCE_ON True

python train_odin.py \
    --dist-url=tcp://127.0.0.1:8474 \
    --num-gpus 1 \
    --resume \
    --config-file configs/scannet_context/swin_3d.yaml \
    --eval-only \
    OUTPUT_DIR outputs/mast3r_without_segment \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    TEST.EVAL_PERIOD 4000 \
    INPUT.FRAME_LEFT 9 \
    INPUT.FRAME_RIGHT 9 \
    INPUT.SAMPLING_FRAME_NUM 19 \
    MODEL.WEIGHTS checkpoints/scannet_swin_50.0_64k_6k.pth \
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
    DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k',)" \
    DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k')" \
    MODEL.PIXEL_DECODER_PANET True \
    MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
    SKIP_CLASSES "[19, 20]" \
    USE_GHOST_POINTS True \
    MODEL.FREEZE_BACKBONE False \
    SOLVER.TEST_IMS_PER_BATCH 1 \
    SAMPLING_STRATEGY consecutive \
    USE_SEGMENTS False \
    SOLVER.MAX_ITER 100000 \
    DATALOADER.NUM_WORKERS 8 \
    DATALOADER.TEST_NUM_WORKERS 2 \
    MAX_FRAME_NUM -1 \
    MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
    MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
    USE_WANDB False \
    USE_MLP_POSITIONAL_ENCODING True \
    HIGH_RES_INPUT False \
    EVAL_PER_IMAGE False \
    DEPTH_PREFIX depth_mast3r \
    MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
    MODEL.MASK_FORMER.TEST.INSTANCE_ON True    


# python train_odin.py \
#     --dist-url=tcp://127.0.0.1:8474 \
#     --num-gpus 1 \
#     --resume \
#     --config-file configs/scannet_context/swin_3d.yaml \
#     --eval-only \
#     OUTPUT_DIR outputs/mast3r_without_segment \
#     SOLVER.IMS_PER_BATCH 4 \
#     SOLVER.CHECKPOINT_PERIOD 4000 \
#     TEST.EVAL_PERIOD 4000 \
#     INPUT.FRAME_LEFT 9 \
#     INPUT.FRAME_RIGHT 9 \
#     INPUT.SAMPLING_FRAME_NUM 19 \
#     MODEL.WEIGHTS checkpoints/scannet_swin_50.0_64k_6k.pth \
#     SOLVER.BASE_LR 1e-4 \
#     INPUT.IMAGE_SIZE 256 \
#     MODEL.CROSS_VIEW_CONTEXTUALIZE True \
#     INPUT.CAMERA_DROP True \
#     INPUT.STRONG_AUGS True \
#     INPUT.AUGMENT_3D True \
#     INPUT.VOXELIZE True \
#     INPUT.SAMPLE_CHUNK_AUG True \
#     MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
#     MODEL.CROSS_VIEW_BACKBONE True \
#     DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k',)" \
#     DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k')" \
#     MODEL.PIXEL_DECODER_PANET True \
#     MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
#     MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
#     SKIP_CLASSES "[19, 20]" \
#     USE_GHOST_POINTS False \
#     MODEL.FREEZE_BACKBONE False \
#     SOLVER.TEST_IMS_PER_BATCH 1 \
#     SAMPLING_STRATEGY consecutive \
#     USE_SEGMENTS False \
#     SOLVER.MAX_ITER 100000 \
#     DATALOADER.NUM_WORKERS 8 \
#     DATALOADER.TEST_NUM_WORKERS 2 \
#     MAX_FRAME_NUM -1 \
#     MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
#     MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
#     USE_WANDB False \
#     USE_MLP_POSITIONAL_ENCODING True \
#     HIGH_RES_INPUT True \
#     EVAL_PER_IMAGE True \
#     DEPTH_PREFIX depth_mast3r \
#     MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
#     MODEL.MASK_FORMER.TEST.INSTANCE_ON True        
