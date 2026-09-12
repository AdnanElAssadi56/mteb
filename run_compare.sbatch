#!/bin/bash
#SBATCH --job-name=moeb-cmp
#SBATCH --partition=guest
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/cmp_%j.out
#SBATCH --error=logs/cmp_%j.err
MODEL="$1"; shift
cd /data/home/niklas/adnan/mteb-moeb
export LD_LIBRARY_PATH=/data/home/niklas/adnan/mteb/.ffmpeg/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "=== $SLURM_JOB_ID model=$MODEL $(date -Iseconds)"
.venv-gpu/bin/python scripts/moeb_run.py --model "$MODEL" --device cuda \
    --batch-size 8 --num-proc 4 --offline --overwrite --tasks "$@"
echo "=== exit=$? $(date -Iseconds)"
