#!/bin/bash
# Sequential 80-epoch training queue for the Stage1 candidate experiments.
#
# The report requires one variable at a time and the paper's training recipe
# (4 GPUs, batch 2 per GPU = 8 global, 80 epochs, fixed seed), so the runs are
# serialised on all four GPUs.  The script waits for the currently running
# experiment to finish first, then runs the candidates in priority order:
#
#   1. vtsample   - sampling view transformation (report step 4)
#   2. radaronly  - trained radar-only control (report step 2 caveat)
#   3. vodanchors - anchors fitted to the measured VoD statistics (step 5)
#
# Each run writes its own log; the per-epoch evaluation written by train.py
# lands in <run>/eval/eval_during_train and the last ten epochs are re-evaluated
# in <run>/eval/eval_with_train for the official re-scoring step.
set +u

REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
cd "$REPO"
# shellcheck disable=SC1091
set +u
source /home/tjh2026/anaconda3/etc/profile.d/conda.sh
conda activate cvfusion-paper

# The bash tool runs every call inside its own PID namespace, so pgrep cannot
# see a training job started by an earlier call.  Wait for a file marker
# instead: train.py writes one result.pkl per evaluated epoch after the run
# finishes, and epoch 80 of the post-training evaluation is the last one.
WAIT_MARKER="$REPO/output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1/eval/eval_with_train/epoch_80/val/result.pkl"
echo "[$(date)] queue started, waiting for $WAIT_MARKER" >> "$AUDIT/queue.log"
while [ ! -f "$WAIT_MARKER" ]; do
  sleep 60
done
echo "[$(date)] imgmulti finished, starting the queue" >> "$AUDIT/queue.log"

run_one () {
  local cfg="$1" tag="$2" log="$3"
  echo "[$(date)] starting $tag ($cfg)" >> "$AUDIT/queue.log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py \
    --launcher pytorch \
    --cfg_file "$cfg" \
    --batch_size 8 \
    --epochs 80 \
    --workers 2 \
    --ckpt_save_interval 1 \
    --max_ckpt_save_num 30 \
    --extra_tag "$tag" > "$log" 2>&1
  echo "[$(date)] $tag finished with exit code $?" >> "$AUDIT/queue.log"
  sleep 30
}

run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vtsample.yaml \
        paper_stage1_vtsample_v1 "$AUDIT/train_vtsample.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_radaronly.yaml \
        paper_stage1_radaronly_v1 "$AUDIT/train_radaronly.log"
run_one tools/cfgs/vod_models/vod_cvfusion_paper_stage1_vodanchors.yaml \
        paper_stage1_vodanchors_v1 "$AUDIT/train_vodanchors.log"

echo "[$(date)] queue complete" >> "$AUDIT/queue.log"
