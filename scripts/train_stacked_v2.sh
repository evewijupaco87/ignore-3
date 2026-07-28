#!/usr/bin/env bash
# Train the v2 stacked Poker44 model for live validator reward.
#
# Default: AP-first calibration (ranking quality), isotonic stack calibrator,
# score_remap enabled, and no post-remap logit shift.
# Leaderboard pattern: high AP + low recall + FPR << 10% -> reward ~0.55-0.60.
#
# Usage:
#   ./scripts/train_stacked_v2.sh
#   OUTPUT=models/poker44_stacked_robust.joblib ./scripts/train_stacked_v2.sh
#   HOLDOUT_SOURCE_DATES=2026-05-08 EXCLUDE_TRAIN_SOURCE_DATES=2026-05-07 ./scripts/train_stacked_v2.sh
#
# Piecewise sequence LR (epochs 1-4 at 1.3e-3, then 5-8 at 1e-3):
#   SEQUENCE_EPOCHS=8 SEQUENCE_LEARNING_RATE_SCHEDULE="1.3e-3:4,1e-3:4" ./scripts/train_stacked_v2.sh
#
# Legacy full-feature + score_remap training:
#   ROBUST_FEATURES_ONLY=0 NO_SCORE_REMAP=0 ./scripts/train_stacked_v2.sh
#
# After training, deploy and check live calibration:
#   POKER44_MODEL_PATH=$(pwd)/models/poker44_stacked_robust.joblib \
#   POKER44_LOG_SCORE_COMPONENTS=1 pm2 restart wolf_miner_5 --update-env
#   sleep 360
#   python -m training.diagnose_live_scores --log ~/.pm2/logs/wolf-miner-5-out.log --last 1

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="${OUTPUT:-models/poker44_stacked_robust.joblib}"
BENCHMARK_PATH="${BENCHMARK_PATH:-}"
# Track whether the caller explicitly set HOLDOUT_SOURCE_DATES (even to "")
# BEFORE applying the legacy default below -- needed so USE_RELEASED_SPLIT=1
# without an explicit holdout doesn't silently inherit the stale "2026-05-08"
# default (which predates the current benchmark files and would error out).
_HOLDOUT_SOURCE_DATES_EXPLICIT="${HOLDOUT_SOURCE_DATES+1}"
HOLDOUT_SOURCE_DATES="${HOLDOUT_SOURCE_DATES:-}"
EXCLUDE_TRAIN_SOURCE_DATES="${EXCLUDE_TRAIN_SOURCE_DATES:-}"
TARGET_FPR="${TARGET_FPR:-0.04}"
MAX_VALIDATOR_FPR="${MAX_VALIDATOR_FPR:-0.05}"
CALIBRATION_OBJECTIVE="${CALIBRATION_OBJECTIVE:-ap_first}"
STACK_CALIBRATOR="${STACK_CALIBRATOR:-isotonic}"
ISOTONIC_CALIBRATION_BLEND="${ISOTONIC_CALIBRATION_BLEND:-0.5}"
HUMAN_WEIGHT="${HUMAN_WEIGHT:-1.3}"
META_C="${META_C:-1.0}"
N_FOLDS="${N_FOLDS:-5}"          # set to 1 for single holdout split (no k-fold)
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.20}"  # val fraction when N_FOLDS=1
SEED="${SEED:-42}"
MAX_FEATURES="${MAX_FEATURES:-0}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.10}"
ROBUST_FEATURES_ONLY="${ROBUST_FEATURES_ONLY:-1}"
NO_SCORE_REMAP="${NO_SCORE_REMAP:-0}"
NO_SCORE_LOGIT_TUNE="${NO_SCORE_LOGIT_TUNE:-1}"

if [[ "${USE_RELEASED_SPLIT:-0}" == "1" && -z "$_HOLDOUT_SOURCE_DATES_EXPLICIT" ]]; then
  # Plain released-split mode: no holdout dates were explicitly requested, so
  # don't let the legacy HOLDOUT_SOURCE_DATES default drag in an unrelated
  # (and likely absent-from-the-file) date.
  HOLDOUT_SOURCE_DATES=""
fi

EXTRA_ARGS=()
if [[ -n "$HOLDOUT_SOURCE_DATES" ]]; then
  EXTRA_ARGS+=(--holdout-source-dates "$HOLDOUT_SOURCE_DATES")
fi
if [[ "${USE_RELEASED_SPLIT:-0}" == "1" ]]; then
  # Uses the benchmark's own per-date train/validation split field. Alone:
  # trains on 'train'-tagged, evals on 'validation'-tagged, across ALL dates
  # (CAVEAT: train/validation for the same date share ~65-90% identical hand
  # content, so this is NOT leakage-free the way HOLDOUT_SOURCE_DATES is --
  # expect a different, not necessarily optimistic, number; see code comment
  # on _split_released). Combined with HOLDOUT_SOURCE_DATES: trains on ALL
  # tags from non-holdout dates, evals ONLY on 'validation'-tagged holdout-date
  # examples, and discards 'train'-tagged holdout-date examples entirely --
  # this combination IS date-disjoint / leakage-safe.
  EXTRA_ARGS+=(--use-released-split)
fi
if [[ -n "$EXCLUDE_TRAIN_SOURCE_DATES" ]]; then
  EXTRA_ARGS+=(--exclude-train-source-dates "$EXCLUDE_TRAIN_SOURCE_DATES")
fi
if [[ "${PER_SOURCE_DATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--per-source-date)
fi
if [[ "${DISABLE_LIGHTGBM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-lightgbm)
fi
if [[ "${DISABLE_XGBOOST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-xgboost)
fi
if [[ "${DISABLE_CATBOOST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-catboost)
fi
if [[ "${DISABLE_EXTRATREES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-extratrees)
fi
if [[ "${DISABLE_RANDOMFOREST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-randomforest)
fi
if [[ "${ENABLE_GPU_TREES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-gpu-trees)
fi
if [[ "${ENABLE_SEQUENCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-sequence)
  EXTRA_ARGS+=(--sequence-epochs "${SEQUENCE_EPOCHS:-4}")
  EXTRA_ARGS+=(--sequence-batch-size "${SEQUENCE_BATCH_SIZE:-32}")
  EXTRA_ARGS+=(--sequence-learning-rate "${SEQUENCE_LEARNING_RATE:-1e-3}")
  if [[ -n "${SEQUENCE_LEARNING_RATE_SCHEDULE:-}" ]]; then
    EXTRA_ARGS+=(--sequence-learning-rate-schedule "$SEQUENCE_LEARNING_RATE_SCHEDULE")
  fi
  EXTRA_ARGS+=(--sequence-d-model "${SEQUENCE_D_MODEL:-64}")
  EXTRA_ARGS+=(--sequence-heads "${SEQUENCE_HEADS:-4}")
  EXTRA_ARGS+=(--sequence-action-layers "${SEQUENCE_ACTION_LAYERS:-2}")
  EXTRA_ARGS+=(--sequence-hand-layers "${SEQUENCE_HAND_LAYERS:-1}")
  EXTRA_ARGS+=(--sequence-max-hands "${SEQUENCE_MAX_HANDS:-64}")
  EXTRA_ARGS+=(--sequence-max-actions "${SEQUENCE_MAX_ACTIONS:-12}")
  EXTRA_ARGS+=(--sequence-dropout "${SEQUENCE_DROPOUT:-0.15}")
  EXTRA_ARGS+=(--sequence-device "${SEQUENCE_DEVICE:-cuda}")
fi
if [[ "${SEQUENCE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--sequence-only)
fi
if [[ "${NO_MINER_VISIBLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-miner-visible-payload)
fi
if [[ "$ROBUST_FEATURES_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--robust-features-only)
fi
if [[ "$NO_SCORE_REMAP" == "1" ]]; then
  EXTRA_ARGS+=(--no-score-remap)
else
  EXTRA_ARGS+=(--score-remap-temperature-grid "${SCORE_REMAP_TEMPERATURE_GRID:-0.08,0.10,0.12,0.18,0.25,0.35,0.50,0.65,0.85,1.0}")
fi
if [[ "${NO_SCORE_REMAP_PREFER_SMOOTH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-score-remap-prefer-smooth)
fi
# Fixed (not fitted) score_remap threshold -- robust to the benchmark->live date
# shift under the 0.1.34 reward. Needs NO_SCORE_REMAP=0. Typical: 0.70.
if [[ -n "${FIXED_SCORE_REMAP_THRESHOLD:-}" ]]; then
  EXTRA_ARGS+=(--fixed-score-remap-threshold "$FIXED_SCORE_REMAP_THRESHOLD")
  EXTRA_ARGS+=(--fixed-score-remap-temperature "${FIXED_SCORE_REMAP_TEMPERATURE:-0.25}")
fi
if [[ "$NO_SCORE_LOGIT_TUNE" == "1" ]]; then
  EXTRA_ARGS+=(--no-score-logit-tune)
fi
# Meta hard-bot reweighting (targets recall@tight-FPR). Env-gated; defaults to
# trainer defaults (2.5/2.0) when unset.
if [[ -n "${META_HARD_BOT_WEIGHT:-}" ]]; then
  EXTRA_ARGS+=(--meta-hard-bot-weight "$META_HARD_BOT_WEIGHT")
fi
if [[ -n "${META_HARD_BOT_GAMMA:-}" ]]; then
  EXTRA_ARGS+=(--meta-hard-bot-gamma "$META_HARD_BOT_GAMMA")
fi
EXTRA_ARGS+=(--calibration-objective "$CALIBRATION_OBJECTIVE")
EXTRA_ARGS+=(--stack-calibrator "$STACK_CALIBRATOR")
EXTRA_ARGS+=(--isotonic-calibration-blend "$ISOTONIC_CALIBRATION_BLEND")

mkdir -p "$(dirname "$OUTPUT")" logs

python -m training.train_model_v2 \
  --output "$OUTPUT" \
  ${BENCHMARK_PATH:+--benchmark-path "$BENCHMARK_PATH"} \
  --holdout-source-dates "$HOLDOUT_SOURCE_DATES" \
  --exclude-train-source-dates "$EXCLUDE_TRAIN_SOURCE_DATES" \
  --target-fpr "$TARGET_FPR" \
  --max-validator-fpr "$MAX_VALIDATOR_FPR" \
  --calibration-fraction "$CALIBRATION_FRACTION" \
  --human-weight-multiplier "$HUMAN_WEIGHT" \
  --meta-c "$META_C" \
  --n-folds "$N_FOLDS" \
  --holdout-frac "$HOLDOUT_FRAC" \
  --seed "$SEED" \
  --max-features "$MAX_FEATURES" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/train_$(basename "$OUTPUT" .joblib)_$(date +%Y%m%d_%H%M%S).log"
