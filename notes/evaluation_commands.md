# Evaluation Commands

## Step 1 — GPU check
nvidia-smi

## Step 2 — Small verification
CUDA_VISIBLE_DEVICES=0 python my_crs/evaluate.py --max_samples 3 --format 3 --dataset redial --recommendation_only

## Step 3 — Medium verification
CUDA_VISIBLE_DEVICES=0 python my_crs/evaluate.py --max_samples 50 --format 3 --dataset redial --recommendation_only

## Step 4 — Official recommendation evaluation
CUDA_VISIBLE_DEVICES=0 python my_crs/evaluate.py --max_samples 200 --format 3 --dataset redial --recommendation_only

## Step 5 — Full response evaluation, optional
CUDA_VISIBLE_DEVICES=0 python my_crs/evaluate.py --max_samples 50 --format 3 --dataset redial
