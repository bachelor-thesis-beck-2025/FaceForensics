#!/bin/bash

PYTHONPATH='.':$PYTHONPATH \
python evaluate.py \
--images_path '/mnt/md0/beck/datasets/benchmarks/test/' \
--model_path /home/beck/repos/FaceForensics/faceforensics++_models_subset/face_detection/xception/all_c23.p \
--cuda \
--skip_no_face \
--output_csv results/results_evaluation.csv \
--root_folder
