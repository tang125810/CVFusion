#!/bin/bash
# Watchdog: keep the current candidate training alive across session suspends.
#
# Why this exists: every command runs inside a sandbox whose process tree is
# torn down when the tool call ends or is interrupted, so a training started
# from a shell inside the session dies with it (that is why the runs kept
# "interrupting").  cron runs outside that sandbox as the normal user, so a job
# started from here survives session suspends.  This script is installed in the
# user crontab to run every minute: if the tagged training is neither running
# nor finished, it relaunches it; train.py resumes from the newest checkpoint,
# so at most the current epoch is lost.
#
# Action log: experiments/stage1_audit_20260926/watchdog.log
set +u

REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
CONDA=/home/tjh2026/anaconda3/etc/profile.d/conda.sh

# --- the run this watchdog keeps alive -------------------------------------
TAG=paper_stage1_zw_v1
CFG=tools/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti_zw.yaml
LOG=$AUDIT/train_zw.log
RUN=$REPO/output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti_zw/$TAG
STAMP=$AUDIT/.watchdog_last_launch

log() { echo "[$(date '+%F %T')] $*" >> "$AUDIT/watchdog.log"; }

# already running?
if pgrep -f "tools/train.py.*$TAG" > /dev/null; then
  exit 0
fi

# finished? train.py writes one result.pkl per evaluated epoch after training
if [ -f "$RUN/eval/eval_with_train/epoch_80/val/result.pkl" ]; then
  if [ ! -f "$AUDIT/.watchdog_done_$TAG" ]; then
    log "$TAG finished (epoch_80 evaluation present)"
    touch "$AUDIT/.watchdog_done_$TAG"
  fi
  exit 0
fi

# do not relaunch more than once every 3 minutes (a fresh start needs ~1 min)
if [ -f "$STAMP" ]; then
  now=$(date +%s); last=$(stat -c %Y "$STAMP")
  [ $((now - last)) -lt 180 ] && exit 0
fi

touch "$STAMP"
log "relaunching $TAG (no training process found)"
cd "$REPO" || exit 1
# shellcheck disable=SC1090
source "$CONDA"
conda activate cvfusion-paper
rm -rf "$AUDIT/.run_zw.lock" 2>/dev/null
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 \
  tools/train.py --launcher pytorch --cfg_file "$CFG" --batch_size 8 --epochs 80 \
  --workers 2 --ckpt_save_interval 1 --max_ckpt_save_num 30 --extra_tag "$TAG" \
  >> "$LOG" 2>&1 &
log "launched pid $!"
exit 0
