#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="list"
GPUS=1
WORK_ROOT="work_dirs/micro_head_ablation"
DISABLE_PRETRAINED=1
RUN_FILTER=""
TRAIN_EXTRA_ARGS=()

ABLA_NAMES=(
  baseline
  micro_base
  stage1_only
  no_infer_enhance
  levels_p2_only
  context_on
  area256
  area2304
  radius0
  radius2
  enhance025
  enhance075
  feat64
  feat192
)

ABLA_CONFIGS=(
  local_configs/ablation_micro_head_convnext_t_baseline.py
  local_configs/ablation_micro_head_convnext_t_micro_base.py
  local_configs/ablation_micro_head_convnext_t_stage1_only.py
  local_configs/ablation_micro_head_convnext_t_no_infer_enhance.py
  local_configs/ablation_micro_head_convnext_t_levels_p2_only.py
  local_configs/ablation_micro_head_convnext_t_context_on.py
  local_configs/ablation_micro_head_convnext_t_area256.py
  local_configs/ablation_micro_head_convnext_t_area2304.py
  local_configs/ablation_micro_head_convnext_t_radius0.py
  local_configs/ablation_micro_head_convnext_t_radius2.py
  local_configs/ablation_micro_head_convnext_t_enhance025.py
  local_configs/ablation_micro_head_convnext_t_enhance075.py
  local_configs/ablation_micro_head_convnext_t_feat64.py
  local_configs/ablation_micro_head_convnext_t_feat192.py
)

usage() {
  cat <<'EOF'
Usage:
  bash tools/misc/run_micro_head_ablation.sh --mode list
  bash tools/misc/run_micro_head_ablation.sh --mode build
  bash tools/misc/run_micro_head_ablation.sh --mode smoke --gpus 1
  bash tools/misc/run_micro_head_ablation.sh --mode train --gpus 2

Options:
  --mode <list|build|smoke|train>
  --gpus <int>                 Use dist_train.sh when gpus > 1 for smoke/train.
  --work-root <path>           Root directory for generated work_dirs.
  --run <csv names>            Run only a subset, e.g. micro_base,area256.
  --keep-pretrained            Do not inject model.backbone.init_cfg=None.
  -- <extra args>              Extra args passed through to train.py/dist_train.sh.

Available run names:
  baseline,micro_base,stage1_only,no_infer_enhance,levels_p2_only,context_on,
  area256,area2304,radius0,radius2,enhance025,enhance075,feat64,feat192
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --run)
      RUN_FILTER="$2"
      shift 2
      ;;
    --keep-pretrained)
      DISABLE_PRETRAINED=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

select_runs() {
  SELECTED_NAMES=()
  SELECTED_CONFIGS=()

  if [[ -z "$RUN_FILTER" ]]; then
    SELECTED_NAMES=("${ABLA_NAMES[@]}")
    SELECTED_CONFIGS=("${ABLA_CONFIGS[@]}")
    return
  fi

  IFS=',' read -r -a REQUESTED <<< "$RUN_FILTER"
  for requested in "${REQUESTED[@]}"; do
    found=0
    for idx in "${!ABLA_NAMES[@]}"; do
      if [[ "${ABLA_NAMES[$idx]}" == "$requested" ]]; then
        SELECTED_NAMES+=("${ABLA_NAMES[$idx]}")
        SELECTED_CONFIGS+=("${ABLA_CONFIGS[$idx]}")
        found=1
        break
      fi
    done
    if [[ $found -eq 0 ]]; then
      echo "Unknown run name: $requested" >&2
      exit 1
    fi
  done
}

build_cfg_options() {
  CFG_OPTIONS=()
  if [[ $DISABLE_PRETRAINED -eq 1 ]]; then
    CFG_OPTIONS+=(model.backbone.init_cfg=None)
  fi
}

run_build() {
  local cmd=(python tools/misc/check_moe_build.py)
  if [[ $DISABLE_PRETRAINED -eq 1 ]]; then
    cmd+=(--disable-pretrained)
  fi
  cmd+=(--configs)
  cmd+=("${SELECTED_CONFIGS[@]}")
  echo "Running build-only for ${#SELECTED_CONFIGS[@]} configs"
  PYTHONPATH=. "${cmd[@]}"
}

run_train_loop() {
  local iter_overrides=()
  local train_script="tools/train.py"
  local dist_script="tools/dist_train.sh"

  if [[ "$MODE" == "smoke" ]]; then
    iter_overrides=(
      runner.max_iters=1
      data.samples_per_gpu=1
      data.workers_per_gpu=0
      log_config.interval=1
      checkpoint_config.interval=1
    )
  fi

  build_cfg_options
  CFG_OPTIONS+=("${iter_overrides[@]}")

  mkdir -p "$WORK_ROOT"
  for idx in "${!SELECTED_NAMES[@]}"; do
    local name="${SELECTED_NAMES[$idx]}"
    local cfg="${SELECTED_CONFIGS[$idx]}"
    local work_dir="$WORK_ROOT/$name"

    echo "Running $MODE for $name"
    if [[ $GPUS -gt 1 ]]; then
      local cmd=(bash "$dist_script" "$cfg" "$GPUS" --work-dir "$work_dir" --no-validate)
      if [[ ${#CFG_OPTIONS[@]} -gt 0 ]]; then
        cmd+=(--cfg-options "${CFG_OPTIONS[@]}")
      fi
      cmd+=("${TRAIN_EXTRA_ARGS[@]}")
      PYTHONPATH=. "${cmd[@]}"
    else
      local cmd=(python "$train_script" "$cfg" --work-dir "$work_dir" --no-validate)
      if [[ ${#CFG_OPTIONS[@]} -gt 0 ]]; then
        cmd+=(--cfg-options "${CFG_OPTIONS[@]}")
      fi
      cmd+=("${TRAIN_EXTRA_ARGS[@]}")
      PYTHONPATH=. "${cmd[@]}"
    fi
  done
}

run_list() {
  for idx in "${!SELECTED_NAMES[@]}"; do
    echo "${SELECTED_NAMES[$idx]} -> ${SELECTED_CONFIGS[$idx]}"
  done
}

select_runs

case "$MODE" in
  list)
    run_list
    ;;
  build)
    run_build
    ;;
  smoke|train)
    run_train_loop
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac