#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
python detect_from_images.py \
--images_path '/mnt/md0/beck/datasets/benchmarks/faceforensics_benchmark_images/' \
--model_path /home/beck/repos/FaceForensics/faceforensics++_models_subset/face_detection/xception/all_c23.p \
--output_csv results/submission1.csv \
--cuda
