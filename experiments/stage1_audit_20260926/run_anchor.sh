#!/bin/bash
# VoD-fitted anchor candidate (technical report step 5 follow-up).
# Single instance guard; train.py auto-resumes from the newest checkpoint.
set +u
REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
cd "$REPO"
source /home/tjh2026/anaconda3/etc/profile.d/conda.sh
conda activate cvfusion-paper
if ! mkdir "$AUDIT/.run_anchor.lock" 2>/dev/null; then
  echo "[$(date)] run_anchor already active" >> "$AUDIT/queue.log"; exit 0
fi
trap 'rmdir "$AUDIT/.run_anchor.lock" 2>/dev/null' EXIT
echo "[$(date)] run_anchor started" >> "$AUDIT/queue.log"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  tools/train.py --launcher pytorch \
  --cfg_file tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vodanchors.yaml \
  --batch_size 8 --epochs 80 --workers 2 --ckpt_save_interval 1 --max_ckpt_save_num 30 \
  --extra_tag paper_stage1_vodanchors_v1 >> "$AUDIT/train_vodanchors.log" 2>&1
echo "[$(date)] vodanchors exited with $?" >> "$AUDIT/queue.log"
