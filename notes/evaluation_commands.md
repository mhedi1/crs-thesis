# Official Evaluation Commands

## Quick verification
```bash
cd ~/work/crs-thesis
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python my_crs/evaluate.py --max_samples 3 --format 3 --dataset redial --recommendation_only