#!/bin/bash
# 只看当前训练进度：epoch、最新一次评估、是否还在跑
cd "$(dirname "$0")/../.."
LOG=experiments/stage1_audit_20260926/train_hstats.log
echo "时间        $(date '+%F %H:%M:%S')"
echo "日志更新    $(date -r $LOG '+%H:%M:%S')  （若明显早于上面时间，说明训练已中断）"
echo "进度        $(tail -c 500 $LOG | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+.*' | tail -1)"
echo "最新评估    $(grep -a '3d   AP' $LOG | tail -3 | tr '\n' ' ')"
echo "official    $(ls output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti_hstats/paper_stage1_hstats_v1/ckpt/ 2>/dev/null | wc -l) checkpoints saved"
