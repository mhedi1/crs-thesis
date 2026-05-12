# JupyterHub Setup Notes

## Why JupyterHub was needed
The local machine can access the Qwen API only from inside the UJA network. JupyterHub is hosted inside the UJA environment, so it can access:
- GPU resources
- Qwen API at port 8050
- project files after upload

## Environment
- Python: 3.10.12
- Torch: 2.6.0+cu124
- CUDA available: True
- Qwen API test: successful with qwen3.5:35b, response time around 49 seconds

## Files copied manually
Because the KBRD baseline folder and trained files were too large or not tracked properly in Git, I copied:
- KBRD_project_code.zip
- KBRD_saved.zip
- KBRD_redial_data.zip

## Important issue
The GPUs were almost full, causing CUDA out-of-memory errors. CPU mode did not work because the KBRD agent still expects CUDA.

## Correct evaluation condition
Only run official evaluation when:
- KBRD neural model loads successfully
- no fallback warning replaces KBRD
- Qwen API is responding
- skipped instances are low or zero