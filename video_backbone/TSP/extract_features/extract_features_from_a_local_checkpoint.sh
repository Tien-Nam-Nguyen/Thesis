#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

DATA_PATH=../data/yc2/train
METADATA_CSV_FILENAME=../data/yc2/yc2_train_metadata.csv


# LOCAL_CHECKPOINT=../data/yc2/r2plus1d_34-tac_on_kinetics.pth
LOCAL_CHECKPOINT=../../../save/yc2_newModel_mvit_3/model-last.pth
# BACKBONE=r2plus1d_34
BACKBONE=mvit_v2_s

# Choose the stride between clips, e.g. 16 for non-overlapping clips and 1 for dense overlapping clips
STRIDE=16 

# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda:0

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$METADATA_CSV_FILENAME" ]; then
    echo "METADATA_CSV_FILENAME variable is not set."
    echo "We provide metadata CSV files for ActivityNet and THUMOS14 in the data folder."
    exit 1
fi

if [ -z "$LOCAL_CHECKPOINT" ]; then
    echo "LOCAL_CHECKPOINT variable is not set."
    echo "Please set LOCAL_CHECKPOINT to the location of the local checkpoint .pth file."
    echo "Make sure to set the correct BACKBONE variable as well."
    exit 1
fi

if [ -z "$BACKBONE" ]; then
    echo "BACKBONE variable is not set."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################

# OUTPUT_DIR=output/local_checkpoint_${BACKBONE}_features/stride_${STRIDE}/
OUTPUT_DIR=output/

# source activate tsp
mkdir -p $OUTPUT_DIR

python extract_features.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--local-checkpoint $LOCAL_CHECKPOINT \
--backbone $BACKBONE \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $OUTPUT_DIR \
--workers 4 \
