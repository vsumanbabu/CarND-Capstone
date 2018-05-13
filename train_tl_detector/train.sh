# README

# The first attempt was made using the data provided by Shyam Jagannathan 
# download this from at https://drive.google.com/drive/folders/0Bz-TOGv42ojzOHhpaXJFdjdfZTA

# Requirements
# ------------
# This script require Tensorflow Object Detection API follow this guide to install:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# Required structure
# ------------------
# +train_tl_detector
#  - train.sh
#  +data
#   +sim
#   +real
#  +models
#   +model
#    +sim-train
#    +sim-eval
#    +sim-export
#    +site-train
#    +site-eval
#    +site-export

# download model
cd models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -xzvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

export TF_DETECT_PATH = 'REPLACE_WITH_PATH'

# Train Sim Model
# Require patch library: https://github.com/tensorflow/models/issues/3705#issuecomment-375563179
python "${TF_DETECT_PATH}/models/research/object_detection/train.py" \
    --logtostderr \
    --pipeline_config_path=models/ssd_mobilenet_v1_coco_2017_11_17/sim-pipeline.config \
    --train_dir=models/ssd_mobilenet_v1_coco_2017_11_17/sim-train

# Exporting a trained model for inference
python "${TF_DETECT_PATH}/models/research/object_detection/export_inference_graph.py" \
    --input_type image_tensor \
    --pipeline_config_path models/ssd_mobilenet_v1_coco_2017_11_17/sim-pipeline.config \
    --trained_checkpoint_prefix models/ssd_mobilenet_v1_coco_2017_11_17/sim-train/model.ckpt-7358 \
    --output_directory models/ssd_mobilenet_v1_coco_2017_11_17/sim-export

# Train Site Model
python "${TF_DETECT_PATH}/models/research/object_detection/train.py" \
    --logtostderr \
    --pipeline_config_path=models/ssd_mobilenet_v1_coco_2017_11_17/site-pipeline.config \
    --train_dir=models/ssd_mobilenet_v1_coco_2017_11_17/site-train

# Exporting a trained model for inference
python "${TF_DETECT_PATH}/models/research/object_detection/export_inference_graph.py" \
    --input_type image_tensor \
    --pipeline_config_path models/ssd_mobilenet_v1_coco_2017_11_17/site-pipeline.config \
    --trained_checkpoint_prefix models/ssd_mobilenet_v1_coco_2017_11_17/site-train/model.ckpt-7358 \
    --output_directory models/ssd_mobilenet_v1_coco_2017_11_17/site-export