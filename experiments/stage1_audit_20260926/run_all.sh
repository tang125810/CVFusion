#!/bin/bash
# One-job driver for the remaining Stage1 candidate experiments.
#
# Everything runs inside a single background job so that no cross-job
# coordination is needed (the bash tool isolates each call in its own PID
# namespace, so one job cannot see - or wait for - another job's processes).
#
# train.py resumes automatically from the newest checkpoint in the run's ckpt
# directory when --ckpt is not passed, so if this script is killed and
# relaunched it continues where it left off instead of restarting.
#
# Order (technical report section 5):
#   1. imgmulti   - image-feature hypothesis (step 3), currently interrupted at
#                   epoch 44, resumes here
#   2. vtsample   - sampling view transformation (step 4)
#   3. radaronly  - trained radar-only control (step 2 caveat)
#   4. vodanchors - VoD-fitted anchors (step 5 follow-up)
set +u

REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
cd "$REPO"
# shellcheck disable=SC1091
source /home/tjh2026/anaconda3/etc/profile.d/conda.sh
conda activate cvfusion-paper

# single instance guard: mkdir is atomic
if ! mkdir "$AUDIT/.run_all.lock" 2>/dev/null; then
  echo "[$(date)] another run_all.sh is already active, exiting" >> "$AUDIT/queue.log"
  exit 0
fi
trap 'rmdir "$AUDIT/.run_all.lock" 2>/dev/null' EXIT

run_one () {
  local cfg="$1" tag="$2" log="$3"
  echo "[$(date)] starting $tag" >> "$AUDIT/queue.log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py \
    --launcher pytorch \
    --cfg_file "$cfg" \
    --batch_size 8 \
    --epochs 80 \
    --workers 2 \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 30 \
    --extra_tag "$tag" >> "$log" 2>&1
  echo "[$(date)] $tag exited with $?" >> "$AUDIT/queue.log"
  sleep 30
}

echo "[$(date)] run_all started" >> "$AUDIT/queue.log"

# 1. finish imgmulti (resumes from its newest checkpoint)
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti.yaml \
        paper_stage1_imgmulti_v1 "$AUDIT/train_imgmulti.log"
# wait until its post-training evaluation has written every epoch 70..80
MARKER_DIR=output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1/eval/eval_with_train
for e in 70 71 72 73 74 75 76 77 78 79 80; do
  while [ ! -f "$MARKER_DIR/epoch_$e/val/result.pkl" ]; do sleep 60; done
done
echo "[$(date)] imgmulti fully evaluated" >> "$AUDIT/queue.log"

run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vtsample.yaml \
        paper_stage1_vtsample_v1 "$AUDIT/train_vtsample.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_radaronly.yaml \
        paper_stage1_radaronly_v1 "$AUDIT/train_radaronly.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vodanchors.yaml \
        paper_stage1_vodanchors_v1 "$AUDIT/train_vodanchors.log"

echo "[$(date)] run_all complete" >> "$AUDIT/queue.log"
