#!/bin/bash
# 一条命令看训练进度：当前 epoch、同 epoch 曲线对比、GPU、队列状态。
#
#   bash experiments/stage1_audit_20260926/check_progress.sh
#
# 可选：只看某个实验
#   bash experiments/stage1_audit_20260926/check_progress.sh hstats
set +u
REPO=/home/tjh2026/Projects/CVFusion/third_party/RadarPillar
AUDIT=$REPO/experiments/stage1_audit_20260926
cd "$REPO"
source /home/tjh2026/anaconda3/etc/profile.d/conda.sh
conda activate cvfusion-paper

RUNS=(
  "imgmulti|vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1"
  "hstats  |vod_cvfusion_paper_stage1_imgmulti_hstats/paper_stage1_hstats_v1"
  "zvalid  |vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1"
  "vtsample|vod_cvfusion_paper_stage1_vtsample/paper_stage1_vtsample_v1"
  "radaronly|vod_cvfusion_paper_stage1_radaronly/paper_stage1_radaronly_v1"
  "vodanchors|vod_cvfusion_paper_stage1_vodanchors/paper_stage1_vodanchors_v1"
  "caranchor|vod_cvfusion_paper_stage1_caranchor/paper_stage1_caranchor_v1"
  "match3d |vod_cvfusion_paper_stage1_match3d/paper_stage1_match3d_v1"
)
FILTER="${1:-}"

echo "=========== $(date '+%F %H:%M:%S') ==========="
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
echo
echo "--- 正在训练（日志活跃度 / 最新 epoch）---"
for entry in "${RUNS[@]}"; do
  name="${entry%%|*}"; rel="${entry##*|}"
  log="$AUDIT/train_$(echo "$name" | tr -d ' ').log"
  [ -f "$log" ] || continue
  [ -n "$FILTER" ] && [ "$(echo "$name" | tr -d ' ')" != "$FILTER" ] && continue
  mtime=$(date -r "$log" '+%H:%M')
  epoch=$(tail -c 400 "$log" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+' | tail -1)
  ckpt=$(ls "$REPO/output/cfgs/vod_models/$rel/ckpt/" 2>/dev/null | wc -l)
  printf '  %-10s 日志最后写入 %s | 进度 %-22s | checkpoint %s 个\n' "$(echo "$name" | tr -d ' ')" "$mtime" "${epoch:-（无）}" "$ckpt"
done
echo
echo "--- 同 epoch 曲线（训练期项目内口径，3D AP_R11 mAP）---"
for entry in "${RUNS[@]}"; do
  name="$(echo "${entry%%|*}" | tr -d ' ')"; rel="${entry##*|}"
  [ -n "$FILTER" ] && [ "$name" != "$FILTER" ] && continue
  glob="$REPO/output/cfgs/vod_models/$rel/log_train_*.txt"
  ls $glob >/dev/null 2>&1 || continue
  python "$AUDIT/parse_train_curve.py" --logs "$glob" --labels "$name" 2>/dev/null | grep -vE "0 evaluations"
done
echo
echo "--- 队列 ---"
tail -5 "$AUDIT/queue.log" 2>/dev/null
echo
echo "--- 官方口径已评分结果（epochs_*.json 中的最优）---"
python - <<'PY'
import glob, json, os
for f in sorted(glob.glob('/home/tjh2026/Projects/CVFusion/third_party/RadarPillar/experiments/stage1_audit_20260926/epochs_*.json')):
    try:
        rows=[r for r in json.load(open(f))['rows'] if 'entire_area' in r.get('regions',{})]
    except Exception:
        continue
    if not rows: continue
    best=max(rows, key=lambda r: r['regions']['entire_area']['mAP']['3d_r11'])
    ent=best['regions']['entire_area']
    print('  %-34s ep%-3d entire %6.3f (Car %6.2f Ped %6.2f Cyc %6.2f) roi %6.3f'%(
        os.path.basename(f).replace('epochs_','').replace('_exact.json',''), best['epoch'],
        ent['mAP']['3d_r11'], ent['Car']['3d_r11'], ent['Pedestrian']['3d_r11'], ent['Cyclist']['3d_r11'],
        best['regions'].get('roi',{}).get('mAP',{}).get('3d_r11', float('nan'))))
PY
