"""Live validator-query capture (operational, local-only, gitignored).

Persists the UNLABELED chunks validators send at inference = the real live
distribution, for unsupervised domain-adaptation / OOD diagnosis of the
benchmark->live gap. Captures INPUTS ONLY (plus this miner's own score); a live
query carries no ground-truth bot/human label, so nothing written here can serve
as a supervised training label.

Safety contract:
  * OFF by default. Enable with env POKER44_CAPTURE=1.
  * Size-capped per file (POKER44_CAPTURE_MAX_BYTES, default 250MB).
  * Thread-safe (append under a lock) and FAIL-SAFE: every path is wrapped so a
    capture error can never affect serving / scoring.
  * Output is gitignored and never leaves the box.

ATTESTATION: while these captures are used only for diagnosis they do NOT change
your training-data statement. The moment you feed them into training (even
unlabeled, for domain adaptation), update POKER44_MODEL_PRIVATE_DATA_ATTESTATION
truthfully.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

_LOCK = threading.Lock()
# Shared capture dir at the GPU_projects/Poker44 root (parents[2] of this file:
# .../Poker44/Poker44_v1/poker44_ml/live_capture.py -> .../Poker44), so every
# co-located miner/repo writes to one place. Override with POKER44_CAPTURE_DIR.
_DIR = Path(
    os.getenv("POKER44_CAPTURE_DIR")
    or Path(__file__).resolve().parents[2] / "live_capture"
)
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))
# Per-process state: resolved output path + a latch once the size cap is hit.
# _seen holds chunk-content hashes already on disk: validators resend the SAME
# daily snapshot every query (measured: 3900 records = 21 unique chunks), so
# without dedupe the size cap fills with duplicates in hours.
_state: dict[str, Any] = {"path": None, "full": False, "seen": None}


def _chunk_key(chunk: Sequence[dict]) -> str:
    # Sanitized LIVE hands carry NO hand_id, so we must key on full chunk
    # CONTENT (deterministic across a snapshot's re-sends, distinct across
    # different chunks). Keying on hand_id collapsed each 100-chunk snapshot to
    # one-per-size (~21) and silently dropped the rest.
    import hashlib
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_seen(path: Path) -> set:
    seen: set = set()
    try:
        if path.exists():
            with open(path) as handle:
                for line in handle:
                    try:
                        seen.add(_chunk_key(json.loads(line).get("chunk") or []))
                    except Exception:
                        continue
    except Exception:
        pass
    return seen


def enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "0") == "1"


def capture(
    chunks: Sequence[Sequence[dict]],
    scores: Sequence[float],
    miner_id: Any,
    validator: Any,
) -> None:
    """Append one JSONL record per chunk: {t, v, uid, n, score, chunk}.

    Input-only (no labels). Never raises — capture must not affect serving.
    """
    if not enabled() or _state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(exist_ok=True)
        if _state["path"] is None:
            _state["path"] = _DIR / f"capture_{str(miner_id)[:16]}.jsonl"
        path: Path = _state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _state["full"] = True
            return
        if _state["seen"] is None:
            _state["seen"] = _load_seen(path)
        seen: set = _state["seen"]
        ts = round(time.time(), 2)
        vtag = str(validator or "")[:8]
        uid = str(miner_id)
        lines = []
        for chunk, score in zip(chunks, scores):
            key = _chunk_key(chunk)
            if key in seen:
                continue  # duplicate of an already-captured chunk (same snapshot)
            seen.add(key)
            try:
                s = round(float(score), 5)
            except (TypeError, ValueError):
                s = None
            lines.append(
                json.dumps(
                    {"t": ts, "v": vtag, "uid": uid, "n": len(chunk), "score": s, "chunk": chunk},
                    separators=(",", ":"),
                )
            )
        if not lines:
            return
        payload = "\n".join(lines) + "\n"
        with _LOCK:
            with open(path, "a") as handle:
                handle.write(payload)
    except Exception:
        # Capture must NEVER affect serving.
        pass


# ---- batch-level capture: the FULL 100-chunk query saved as ONE record, to a
# SEPARATE file (batch_<uid>.jsonl). Gated by POKER44_CAPTURE_BATCH=1, deduped by
# whole-batch content, independent of the per-chunk capture above. ------------
_batch: dict[str, Any] = {"path": None, "seen": None, "full": False}


def batch_enabled() -> bool:
    return os.getenv("POKER44_CAPTURE_BATCH", "0") == "1"


def _batch_key(chunks) -> str:
    blob = json.dumps(chunks, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_batch_seen(path: Path) -> set:
    seen: set = set()
    try:
        if path.exists():
            with open(path) as handle:
                for line in handle:
                    try:
                        seen.add(_batch_key(json.loads(line).get("chunks") or []))
                    except Exception:
                        continue
    except Exception:
        pass
    return seen


def capture_batch(chunks, scores, miner_id, validator) -> None:
    """Append the whole query batch (all chunks + scores) as one JSON record to
    <dir>/batch_<uid>.jsonl. One record per UNIQUE snapshot. Never raises."""
    if not batch_enabled() or _batch["full"] or not chunks:
        return
    try:
        _DIR.mkdir(exist_ok=True)
        if _batch["path"] is None:
            _batch["path"] = _DIR / f"batch_{str(miner_id)[:16]}.jsonl"
        path: Path = _batch["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _batch["full"] = True
            return
        if _batch["seen"] is None:
            _batch["seen"] = _load_batch_seen(path)
        bkey = _batch_key(chunks)
        if bkey in _batch["seen"]:
            return  # this exact 100-chunk snapshot already saved
        _batch["seen"].add(bkey)
        out_scores = []
        for s in scores:
            try:
                out_scores.append(round(float(s), 6))
            except (TypeError, ValueError):
                out_scores.append(None)
        rec = {
            "t": round(time.time(), 2),
            "v": str(validator or "")[:8],
            "uid": str(miner_id),
            "n_chunks": len(chunks),
            "sizes": [len(c) for c in chunks],
            "scores": out_scores,
            "chunks": list(chunks),
        }
        payload = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        with _LOCK:
            with open(path, "a") as handle:
                handle.write(payload)
    except Exception:
        pass
