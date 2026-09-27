#!/bin/bash
# Remaining Stage1 experiments after the vtsample candidate was rejected early
# (it trailed the baseline at every matched epoch from 10 to 56).
#   1. radaronly  - trained radar-only control (diagnostic, report step 2/5)
#   2. vodanchors - anchors fitted to the measured VoD statistics (step 5)
set +u
REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
cd "$REPO"
source /home/tjh2026/anaconda3/etc/profile.d/conda.sh
conda activate cvfusion-paper
if ! mkdir "$AUDIT/.run_rest.lock" 2>/dev/null; then
  echo "[$(date)] run_rest already active" >> "$AUDIT/queue.log"; exit 0
fi
trap 'rmdir "$AUDIT/.run_rest.lock" 2>/dev/null' EXIT
run_one () {
  echo "[$(date)] starting $2" >> "$AUDIT/queue.log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py --launcher pytorch --cfg_file "$1" --batch_size 8 --epochs 80 \
    --workers 2 --ckpt_save_interval 1 --max_ckpt_save_num 30 --extra_tag "$2" \
    >> "$3" 2>&1
  echo "[$(date)] $2 exited with $?" >> "$AUDIT/queue.log"
  sleep 30
}
echo "[$(date)] run_rest started" >> "$AUDIT/queue.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_radaronly.yaml paper_stage1_radaronly_v1 "$AUDIT/train_radaronly.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vodanchors.yaml paper_stage1_vodanchors_v1 "$AUDIT/train_vodanchors.log"
echo "[$(date)] run_rest complete" >> "$AUDIT/queue.log"
