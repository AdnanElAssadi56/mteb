#!/bin/bash
#SBATCH --job-name=moeb-audio
#SBATCH --partition=guest
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/audio_%x_%j.out
#SBATCH --error=logs/audio_%x_%j.err

MODEL="$1"
SHARD="${2:-0/1}"
cd /data/home/niklas/adnan/mteb-moeb
export LD_LIBRARY_PATH=/data/home/niklas/adnan/mteb/.ffmpeg/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=== $SLURM_JOB_ID  model=$MODEL shard=$SHARD  $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
.venv-gpu/bin/python scripts/moeb_run.py \
    --model "$MODEL" --shard "$SHARD" \
    --device cuda --batch-size 8 --num-proc 4 --offline
echo "=== exit=$? $(date -Iseconds)"
