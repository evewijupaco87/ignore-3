#!/usr/bin/env bash
# Start this Poker44 luck-detector miner under pm2, using the project .env as the
# single source of truth for wallet / hotkey / axon port / model identity, and
# the classic-bittensor virtualenv interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

if [[ ! -f .env ]]; then echo "ERROR: .env not found in $ROOT"; exit 1; fi
set -a; . ./.env; set +a

: "${NETUID:=126}"; : "${NETWORK:=finney}"
: "${WALLET_NAME:?set in .env}"; : "${HOTKEY:?set in .env}"; : "${AXON_PORT:?set in .env}"
: "${PM2_NAME:?set in .env}"; : "${MINER_PYTHON:=/root/miner_venv/bin/python}"

export PYTHONPATH="$ROOT"

ARGS=(--netuid "$NETUID" --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY"
      --subtensor.network "$NETWORK" --axon.port "$AXON_PORT" --logging.debug)
if [[ -n "${ALLOWED_VALIDATOR_HOTKEYS:-}" ]]; then
  read -r -a VHK <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  ARGS+=(--blacklist.allowed_validator_hotkeys "${VHK[@]}")
else
  ARGS+=(--blacklist.force_validator_permit)
fi

pm2 delete "$PM2_NAME" 2>/dev/null || true
pm2 start neurons/miner.py --name "$PM2_NAME" --interpreter "$MINER_PYTHON" \
  --cwd "$ROOT" --max-restarts 50 --restart-delay 5000 -- "${ARGS[@]}"
pm2 save
echo "Started $PM2_NAME (wallet=$WALLET_NAME hotkey=$HOTKEY port=$AXON_PORT)"
