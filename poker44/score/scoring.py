"""Reward and scoring utilities for Poker44 poker bot detection.

This mirrors the **live subnet reward** (Poker44-subnet >= 0.1.34, commit
367a8a2 "Update validator scoring formula", 2026-07-08). The reward is still
rank-led, but it is NO LONGER pure rank-first: the fixed-0.5 operating point
matters again via a ``threshold_sanity`` term plus a hard zero-gate.

    ap_score            = average_precision_score(y_true, y_pred)        # rank
    bot_recall, fpr     = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)   # rank
    threshold_sanity    = quality of the FIXED-0.5 operating point       # NOT rank
    reward = 0.35*ap + 0.30*bot_recall
           + 0.20*threshold_sanity + 0.10*threshold_sanity + 0.05*latency
    # HARD GATE: threshold_sanity == 0  ->  reward == 0

``threshold_sanity`` (``_threshold_metrics``, evaluated at 0.5) is 1.0 when bots
cross 0.5 and the human FPR@0.5 is <= 0.10, 0.0 when no bot chunk crosses 0.5 on
a mixed window (the zero-gate), and decays linearly for FPR@0.5 above 0.10.
``human_safety_penalty`` and ``calibration_quality`` are BOTH aliases of
``threshold_sanity`` (combined weight 0.30). ``latency_quality`` is 1.0 here
(real latency is tracked by the validator separately, not yet in the formula).

Consequence for training: AP and recall@FPR<=0.05 (0.65 of the reward) are
rank-invariant, but calibration / score_remap / the 0.5 crossing now move the
0.30 threshold_sanity term and can trip the zero-gate. Monotonic post-processing
is no longer moot. The pre-0.1.25 fixed-0.5 formula is kept as
:func:`legacy_reward` purely for before/after comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

# Reward term weights (Poker44-subnet 0.1.34, poker44/validator/scoring.py).
AP_WEIGHT = 0.35
BOT_RECALL_WEIGHT = 0.30
HUMAN_SAFETY_WEIGHT = 0.20
CALIBRATION_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded.

    Sweeps every threshold and returns the highest recall whose false-positive
    rate stays <= ``max_fpr`` (and the FPR at that operating point). This is the
    30% recall term of the live reward; it rewards a *clean top of the ranking*
    (catching bots before humans start being flagged), independent of where any
    fixed threshold sits.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def _threshold_metrics(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Quality of the FIXED-``threshold`` operating point (default 0.5).

    Scores are still evaluated rank-first, but they must also be usable as a
    risk threshold. A model that never crosses 0.5 on a mixed labeled window
    cannot operationally flag bots, even if its relative ordering is strong.

    ``threshold_sanity_quality``: 1.0 if bots cross the threshold and the human
    FPR@threshold <= 0.10; 0.0 if a mixed window has no bot crossing (the
    zero-gate); linear decay ``max(0, 1 - (fpr - 0.10) / 0.90)`` for FPR > 0.10.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if scores.size == 0:
        return {
            "hard_bot_recall": 0.0,
            "hard_fpr": 0.0,
            "positive_prediction_rate": 0.0,
            "threshold_sanity_quality": 0.0,
        }

    hard_predictions = scores >= float(threshold)
    positive_prediction_rate = float(np.mean(hard_predictions))
    true_positives = int(np.sum(hard_predictions & (labels == 1)))
    false_positives = int(np.sum(hard_predictions & (labels == 0)))
    hard_bot_recall = (
        true_positives / max(positive_count, 1) if positive_count > 0 else 0.0
    )
    hard_fpr = (
        false_positives / max(negative_count, 1) if negative_count > 0 else 0.0
    )

    if positive_count <= 0 or negative_count <= 0:
        threshold_sanity_quality = 1.0
    elif true_positives <= 0:
        threshold_sanity_quality = 0.0
    elif hard_fpr <= 0.10:
        threshold_sanity_quality = 1.0
    else:
        threshold_sanity_quality = max(0.0, 1.0 - (hard_fpr - 0.10) / 0.90)

    return {
        "hard_bot_recall": float(hard_bot_recall),
        "hard_fpr": float(hard_fpr),
        "positive_prediction_rate": positive_prediction_rate,
        "threshold_sanity_quality": float(threshold_sanity_quality),
    }


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """Live reward (subnet >= 0.1.34). Matches Poker44-subnet ``reward()`` exactly.

    Returns ``(reward, details)``. Keeps the dict keys earlier formulas exposed
    (``fpr``, ``bot_recall``, ``ap_score``, ``human_safety_penalty``,
    ``base_score``, ``reward``) so every caller keeps working, and adds the new
    0.1.34 keys (``calibration_quality``, ``latency_quality``,
    ``threshold_sanity_quality``, ``hard_bot_recall``, ``hard_fpr``,
    ``positive_prediction_rate``).

    * ``ap_score``   — average precision (0.35 term, rank-invariant).
    * ``bot_recall`` — recall at FPR <= 0.05 (0.30 term, rank-invariant).
    * ``human_safety_penalty`` == ``calibration_quality`` == threshold_sanity
      at 0.5 (0.20 + 0.10 = 0.30 combined weight, NOT rank-invariant).
    * zero-gate: threshold_sanity == 0  ->  reward == 0.
    """
    scores = np.asarray(y_pred, dtype=float)
    labels = np.asarray(y_true, dtype=int)

    if scores.size and np.any(labels == 1):
        ap_score = float(average_precision_score(labels, scores))
    else:
        ap_score = 0.0

    bot_recall, fpr = _recall_at_fpr(scores, labels, max_fpr=0.05)
    threshold_metrics = _threshold_metrics(scores, labels, threshold=0.5)
    human_safety_penalty = threshold_metrics["threshold_sanity_quality"]
    calibration_quality = human_safety_penalty
    latency_quality = 1.0

    if human_safety_penalty <= 0:
        base_score = 0.0
        rew = 0.0
    else:
        base_score = (
            AP_WEIGHT * ap_score
            + BOT_RECALL_WEIGHT * bot_recall
            + HUMAN_SAFETY_WEIGHT * human_safety_penalty
            + CALIBRATION_WEIGHT * calibration_quality
            + LATENCY_WEIGHT * latency_quality
        )
        rew = float(np.clip(base_score, 0.0, 1.0))

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "calibration_quality": calibration_quality,
        "latency_quality": latency_quality,
        "base_score": base_score,
        "reward": rew,
        **threshold_metrics,
    }
    return rew, res


def reward_eval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    mode: str = "live",
) -> tuple[float, dict]:
    """Evaluation wrapper around :func:`reward`.

    The 0.1.34 formula has no mode-specific variation implemented, so ``live`` /
    ``base`` / ``soft`` all return the same reward; ``mode`` is retained for CLI
    back-compat and only tags ``reward_mode`` in the returned details.
    """
    if mode not in ("live", "base", "soft"):
        raise ValueError(f"Unknown reward eval mode: {mode!r}")
    rew, details = reward(y_pred, y_true)
    return rew, {**details, "reward_mode": mode}


def format_reward_breakdown(
    ap_score: float,
    bot_recall: float,
    *,
    fpr: float = 0.0,
    reward: float | None = None,
) -> str:
    """One-line decomposition of the live 0.1.34 reward.

    ``reward = 0.35*AP + 0.30*recall@(FPR<=0.05) + [0.30*threshold_sanity@0.5 +
    0.05*latency]``. The bracketed operating-point contribution is recovered
    from the authoritative ``reward`` total (which callers pass in) minus the two
    rank terms, so the printed line always agrees with ``validator_reward``.
    Shows per-term headroom so it is obvious which term to push.
    """
    ap = float(ap_score)
    recall = float(bot_recall)
    ap_term = AP_WEIGHT * ap
    recall_term = BOT_RECALL_WEIGHT * recall
    op_max = HUMAN_SAFETY_WEIGHT + CALIBRATION_WEIGHT + LATENCY_WEIGHT  # 0.35

    gated = False
    if reward is None:
        rew = ap_term + recall_term
        op_term = 0.0
    else:
        rew = float(reward)
        op_term = rew - ap_term - recall_term
        if rew <= 0.0 and (ap_term + recall_term) > 0.0:
            gated = True
            op_term = 0.0

    ap_headroom = AP_WEIGHT * (1.0 - ap)
    recall_headroom = BOT_RECALL_WEIGHT * (1.0 - recall)
    op_headroom = max(0.0, op_max - op_term)
    headrooms = {
        "AP": ap_headroom,
        "recall@FPR<=0.05": recall_headroom,
        "threshold_sanity@0.5": op_headroom,
    }
    push = max(headrooms, key=headrooms.get)

    line = (
        f"reward={rew:.4f} = 0.35*AP({ap:.4f})={ap_term:.4f} "
        f"+ 0.30*recall@FPR<=0.05({recall:.4f}, fpr={fpr:.4f})={recall_term:.4f} "
        f"+ [0.30*threshold_sanity@0.5 + 0.05*latency]={op_term:.4f} | "
        f"headroom AP=+{ap_headroom:.4f} recall=+{recall_headroom:.4f} "
        f"op=+{op_headroom:.4f} -> push {push}"
    )
    if gated:
        line += "  [GATED: threshold_sanity=0 (no bot chunk crossed 0.5) -> reward=0]"
    return line


def legacy_reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """Obsolete pre-0.1.25 reward (fixed-0.5 threshold + FPR-cliff penalty).

    Kept ONLY so you can compare old-vs-new on the same scores. Not wired into
    training or eval. ``reward = (0.65*AP + 0.35*recall@0.5) * (1-fpr)**2`` with
    a hard 0 below FPR 0.10.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    preds = np.round(y_pred).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    negative_count = max(tn + fp, 1)
    positive_count = max(tp + fn, 1)

    fpr = fp / negative_count
    bot_recall = tp / positive_count

    if y_pred.size and np.any(y_true == 1):
        ap_score = float(average_precision_score(y_true, y_pred))
    else:
        ap_score = 0.0

    human_safety_penalty = max(0.0, 1.0 - fpr) ** 2
    if fpr >= 0.10:
        human_safety_penalty = 0.0

    base_score = 0.65 * ap_score + 0.35 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res
