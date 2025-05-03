export output_dir="./res/exp_6/RGB"
export datasets_path="./datasets"
# python train_residualgan.py -cfg ./configs/residualgan.yaml -opts LOSS.DEPTH 0.0 LOSS.DEPTH_CYCLE 0.0 LOSS.ADV 1 LOSS.CYCLE 10 OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False DATASETS.SOURCE_PATH $datasets_path/PotsdamIRRG DATASETS.TARGET_PATH $datasets_path/Vaihingen_dsm &&
# python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir

# python train_residualgan.py -cfg ./configs/residualgan.yaml -opts LOSS.DEPTH 0.0 LOSS.DEPTH_CYCLE 0.0 LOSS.ADV 1 LOSS.CYCLE 10 OUTPUT_DIR $output_dir MODELS.GENERATOR "UNet" MODELS.K_GRAD False DATASETS.SOURCE_PATH $datasets_path/PotsdamRGB DATASETS.TARGET_PATH $datasets_path/Bhopal

# python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir

# python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir

# ***************** OSA False ***************
# python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV False TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir

python train_seg.py -cfg ./configs/segmentation.yaml -opts MODELS.OUT_ADV True TRAIN.BATCH_SIZE 32 DATASETS.EVL_BATCH 32 DATASETS.SOURCE_DATASET_PATH $output_dir/data OUTPUT_DIR $output_dir
