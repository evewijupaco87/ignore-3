"""Poker44 miner with local stacked-model inference and transparent model manifests."""

import asyncio
import hashlib
import logging as stdlogging
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bittensor as bt

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime.
    pass

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    artifact_model_identity,
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

try:
    from poker44_ml.inference import Poker44Model
except ImportError:  # pragma: no cover - optional local-model path.
    Poker44Model = None

from poker44_ml.luck_detector import build_luck_detector

try:
    from poker44_ml import live_capture
except ImportError:  # pragma: no cover - capture is optional.
    live_capture = None

# Strong refs to in-flight fire-and-forget capture tasks (loop keeps only weak
# refs, so without this they can be GC'd mid-write).
_CAPTURE_TASKS: set = set()


class _ScannerNoiseFilter(stdlogging.Filter):
    """Suppress common public-port probe errors emitted before miner routing."""

    _NOISY_SNIPPETS = (
        "UnknownSynapseError",
        "InvalidRequestNameError",
    )
    _NOISY_REQUEST_NAMES = (
        "Synapse name ''",
        "Synapse name 'api'",
        "Synapse name 'mcp'",
        "Synapse name 'jsonrpc'",
        "Synapse name 'robots.txt'",
        "Could not parser request .",
    )

    def filter(self, record: stdlogging.LogRecord) -> bool:
        message = record.getMessage()
        if not any(snippet in message for snippet in self._NOISY_SNIPPETS):
            return True
        if any(name in message for name in self._NOISY_REQUEST_NAMES):
            return False
        return True


class Miner(BaseMinerNeuron):
    """
    Reference miner for the current provider-runtime challenge path.

    This miner scores chunks directly from the incoming hand payloads without
    any local training artifacts. The heuristic emphasizes chunk-level behavior
    consistency, passive regularity, street progression, and showdown tendency.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self._install_scanner_log_filter()
        self.max_hands_per_chunk_eval = max(
            0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
        )
        self.query_log_preview = (
            os.getenv("POKER44_LOG_QUERY_PREVIEW", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.component_debug_logging = (
            os.getenv("POKER44_LOG_SCORE_COMPONENTS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.score_array_logging = (
            os.getenv("POKER44_LOG_SCORE_ARRAYS", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        # Model-external true-rank batch remap (opt-in, default OFF via
        # POKER44_BATCH_RANK). Re-ranks each validator batch so ~target_fraction
        # crosses 0.5 -- per-batch stable and model-agnostic (immune to each
        # model's score scale), which fixes BOTH over-crossing (high FPR@0.5) and
        # under-crossing (zero-gate) under the 0.1.34 reward without retraining.
        # CAVEAT/FLAG: assumes ~balanced batches; on a skewed batch it forces the
        # target fraction across 0.5 regardless of the true bot rate.
        self.batch_rank_enabled = (
            os.getenv("POKER44_BATCH_RANK", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.batch_rank_target_fraction = min(
            0.99, max(0.01, float(os.getenv("POKER44_BATCH_RANK_TARGET_FRACTION", "0.5") or 0.5))
        )
        self.batch_rank_span = min(
            0.98, max(0.10, float(os.getenv("POKER44_BATCH_RANK_SPAN", "0.8") or 0.71))
        )
        if self.batch_rank_enabled:
            bt.logging.info(
                "Batch-rank remap ENABLED (model-external, true-rank) | "
                f"target_fraction={self.batch_rank_target_fraction} "
                f"span={self.batch_rank_span} "
                "(rank-preserving: AP/recall unchanged; sets the 0.5 crossing)"
            )
        self.model_path = Path(
            os.getenv(
                "POKER44_MODEL_PATH",
                str(repo_root / "models" / "poker44_stacked_robust.joblib"),
            )
        )
        self._artifact_identity = artifact_model_identity(self.model_path)
        # Live behavioral backend used when no trained artifact is present. This
        # is the tempo/entropy variant; it also seeds the daily retrain prior.
        self.detector = build_luck_detector()
        self.predictor = None
        self.backend = f"luck-{self.detector.PROFILE}"
        if Poker44Model is not None and self.model_path.exists():
            try:
                self.predictor = Poker44Model(self.model_path)
                self.backend = "benchmark-supervised"
            except Exception as err:
                bt.logging.warning(
                    f"Failed to load local benchmark model at {self.model_path}: {err}. "
                    "Continuing with heuristic backend."
                )

        bt.logging.info(f"🤖 Poker44 Miner started with backend={self.backend}")
        runtime_commit = self._repo_head(repo_root)
        runtime_repo_url = self._normalize_repo_url(self._repo_url(repo_root))
        model_metadata = dict(self.predictor.metadata) if self.predictor is not None else {}
        artifact_repo_commit = str(model_metadata.get("repo_commit", "")).strip()
        artifact_repo_url = self._normalize_repo_url(str(model_metadata.get("repo_url", "")).strip())
        benchmark_rows = int(float(model_metadata.get("benchmark_rows", 0.0) or 0.0))
        ensemble_combiner = str(model_metadata.get("ensemble_combiner", "") or "").strip()
        ensemble_max_blend = model_metadata.get("ensemble_max_blend")
        score_expansion = model_metadata.get("score_expansion") or {}
        score_remap = model_metadata.get("score_remap") or {}
        score_logit_bias = model_metadata.get("score_logit_bias")
        score_logit_temperature = model_metadata.get("score_logit_temperature")
        threshold_calibrator = (
            model_metadata.get("calibrator")
            if isinstance(model_metadata.get("calibrator"), dict)
            else {}
        )
        supervised_notes = (
            "Supervised benchmark model trained on released evaluation chunks"
        )
        artifact_filename = str(
            model_metadata.get("artifact_filename", "")
            or self._artifact_identity.get("artifact_filename", "")
        ).strip()
        if artifact_filename:
            supervised_notes += f"; artifact={artifact_filename}"
        if ensemble_combiner:
            supervised_notes += f"; ensemble_combiner={ensemble_combiner}"
            if ensemble_max_blend is not None:
                supervised_notes += f", ensemble_max_blend={ensemble_max_blend}"
        if score_expansion:
            supervised_notes += f"; score_expansion={score_expansion.get('kind', 'enabled')}"
        if score_remap:
            supervised_notes += (
                f"; score_remap={score_remap.get('kind', 'enabled')} "
                f"threshold={score_remap.get('threshold', 'unknown')}"
            )
        if threshold_calibrator.get("kind") == "threshold_logit_v1":
            supervised_notes += (
                "; threshold_logit_calibration="
                f"threshold={threshold_calibrator.get('threshold', 'unknown')}"
                f", temperature={threshold_calibrator.get('temperature', 'unknown')}"
            )
        if score_logit_bias is not None:
            supervised_notes += (
                f"; score_logit_bias={score_logit_bias}"
                f", score_logit_temperature={score_logit_temperature or 1.0}"
            )
        if threshold_calibrator.get("kind") == "threshold_logit_v1":
            bt.logging.info(
                "Loaded threshold-logit calibration | "
                f"threshold={threshold_calibrator.get('threshold')} "
                f"temperature={threshold_calibrator.get('temperature')} "
                f"human_anchor={threshold_calibrator.get('human_anchor')} "
                f"bot_anchor={threshold_calibrator.get('bot_anchor')} "
                f"human_cutoff={threshold_calibrator.get('human_cutoff')} "
                f"bot_cutoff={threshold_calibrator.get('bot_cutoff')} "
                f"human_quantile={threshold_calibrator.get('human_quantile')} "
                f"bot_quantile={threshold_calibrator.get('bot_quantile')} "
                f"aggregation={threshold_calibrator.get('aggregation')}"
            )
        training_data_statement = self._training_data_statement(
            model_metadata, benchmark_rows=benchmark_rows
        )
        training_data_sources = (
            ["released_training_benchmark"] if self.predictor is not None else ["none"]
        )
        private_data_attestation = (
            "No validator-private data used. Supervised artifacts use "
            "released benchmark labels only."
            if self.predictor is not None
            else "This miner does not train on validator-only evaluation data."
        )
        manifest_notes = (
            supervised_notes
            if self.predictor is not None
            else "Sequence-signature behavioral detector: scores chunk-level action-sequence signature concentration and street-progression uniformity to separate replayed scripted lines from human play."
        )
        if self.predictor is not None and (
            artifact_repo_commit and artifact_repo_commit != runtime_commit
            or artifact_repo_url and artifact_repo_url != runtime_repo_url
        ):
            manifest_notes += (
                f"; training_artifact_repo={artifact_repo_url or 'unknown'}"
                f", training_artifact_commit={artifact_repo_commit or 'unknown'}"
            )
        artifact_sha256 = ""
        if self.predictor is not None and self.model_path.is_file():
            artifact_sha256 = self._sha256_file(self.model_path)
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=self._implementation_files(
                repo_root, has_predictor=self.predictor is not None
            ),
            defaults={
                "model_name": (
                    str(model_metadata.get("model_name", "")).strip()
                    or self._artifact_identity.get("model_name", "")
                    or "poker44_stacked_robust"
                    if self.predictor is not None
                    else "luck-signature-detector"
                ),
                "model_version": (
                    str(model_metadata.get("model_version", "")).strip()
                    or self._artifact_identity.get("model_version", "")
                    or "1"
                    if self.predictor is not None
                    else "3.2.0"
                ),
                "framework": (
                    str(model_metadata.get("framework", "")).strip()
                    or (
                        "stacked-sequence-v2"
                        if model_metadata.get("sequence_enabled")
                        else "stacked-v2"
                    )
                    if self.predictor is not None
                    else "sequence-signature-behavioral-v3"
                ),
                "artifact_filename": (
                    str(model_metadata.get("artifact_filename", "")).strip()
                    or self._artifact_identity.get("artifact_filename", "")
                    if self.predictor is not None
                    else ""
                ),
                "license": "MIT",
                "repo_commit": runtime_commit,
                "repo_url": runtime_repo_url,
                "artifact_url": str(self.model_path.resolve()) if self.predictor is not None else "",
                "artifact_sha256": artifact_sha256,
                "notes": manifest_notes,
                "open_source": True,
                "inference_mode": (
                    "local-joblib" if self.predictor is not None else "heuristic"
                ),
                "training_data_statement": training_data_statement,
                "training_data_sources": training_data_sources,
                "private_data_attestation": private_data_attestation,
                "data_attestation": private_data_attestation,
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        bt.logging.info(f"Axon created: {self.axon}")

    @staticmethod
    def _normalize_repo_url(url: str) -> str:
        cleaned = str(url or "").strip()
        if not cleaned:
            return ""
        if cleaned.startswith("git@"):
            host_path = cleaned.split(":", 1)
            if len(host_path) == 2:
                host = host_path[0][4:]
                path = host_path[1]
                if path.endswith(".git"):
                    path = path[:-4]
                return f"https://{host}/{path}"
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        return cleaned

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _implementation_files(cls, repo_root: Path, *, has_predictor: bool) -> List[Path]:
        files = [Path(__file__).resolve()]
        # The behavioral detector is the live serving backend regardless of
        # whether a trained artifact is loaded, so it is always attested.
        detector_path = repo_root / "poker44_ml" / "luck_detector.py"
        if detector_path.exists():
            files.append(detector_path)
        if not has_predictor:
            return files
        # Serving path: code that actually runs when scoring a chunk.
        serving = (
            "poker44_ml/inference.py",
            "poker44_ml/features.py",
            "poker44_ml/live_capture.py",
            "poker44_ml/sequence_model.py",
            "poker44_ml/stacked.py",
            "poker44_ml/calibration.py",
            "poker44/validator/payload_view.py",
        )
        # Training recipe: not executed at serve time, but included so the
        # published implementation hash also attests how the artifact was
        # produced (reproducibility for the audit lane).
        training = (
            "scripts/train_stacked_v2.sh",
        )
        for relative in (*serving, *training):
            candidate = repo_root / relative
            if candidate.exists():
                files.append(candidate)
        return files

    @staticmethod
    def _training_data_statement(
        model_metadata: Dict[str, Any],
        *,
        benchmark_rows: int,
    ) -> str:
        if not model_metadata:
            return (
                "Reference heuristic miner. No training step. "
                "Uses only runtime chunk features."
            )
        parts = [
            (
                f"Trained on {benchmark_rows} released benchmark chunks with groundTruth labels."
                if benchmark_rows
                else "Trained on released benchmark chunks with groundTruth labels."
            )
        ]
        holdout = model_metadata.get("holdout_source_dates")
        if holdout:
            parts.append(f"Holdout source dates: {holdout}.")
        excluded = model_metadata.get("excluded_train_source_dates")
        if excluded:
            parts.append(f"Excluded train source dates: {excluded}.")
        if model_metadata.get("no_score_remap"):
            parts.append("Post-training score_remap disabled.")
        sequence_config = model_metadata.get("sequence_config")
        if isinstance(sequence_config, dict) and sequence_config:
            epochs = sequence_config.get("n_epochs", sequence_config.get("epochs", "unknown"))
            parts.append(
                "Sequence learner config: "
                f"d_model={sequence_config.get('d_model', 'unknown')}, "
                f"epochs={epochs}, schema_version={sequence_config.get('schema_version', 'unknown')}."
            )
        artifact_filename = str(model_metadata.get("artifact_filename", "")).strip()
        if artifact_filename:
            parts.append(f"Artifact file: {artifact_filename}.")
        return " ".join(parts)

    @staticmethod
    def _repo_head(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _repo_url(repo_root: Path) -> str:
        try:
            completed = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _install_scanner_log_filter() -> None:
        enabled = os.getenv("POKER44_SUPPRESS_SCANNER_ERRORS", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return

        scanner_filter = _ScannerNoiseFilter()
        configured = False

        for handler in getattr(bt.logging, "_handlers", []):
            handler.addFilter(scanner_filter)
            configured = True

        for logger_name in ("bittensor", "uvicorn.access", "uvicorn.error"):
            logger = stdlogging.getLogger(logger_name)
            for handler in logger.handlers:
                handler.addFilter(scanner_filter)
                configured = True

        if configured:
            bt.logging.info("Scanner-noise log filter enabled for invalid public-port probes.")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']} "
            f"policy_violations={self.manifest_compliance.get('policy_violations', [])})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            f"Model path={self.model_path} backend={self.backend}"
        )
        bt.logging.info(
            "Runtime config | "
            f"max_hands_per_chunk_eval={self.max_hands_per_chunk_eval} "
            f"query_log_preview={self.query_log_preview} "
            f"component_debug_logging={self.component_debug_logging} "
            f"score_array_logging={self.score_array_logging}"
        )
        if self.predictor is not None:
            artifact_commit = str(self.predictor.metadata.get("repo_commit", ""))
            runtime_commit = self._repo_head(repo_root)
            bt.logging.info(
                f"Model metadata: feature_count={len(self.predictor.feature_names)} "
                f"framework={self.predictor.metadata.get('framework', 'unknown')} "
                f"artifact_commit={artifact_commit or 'unknown'} "
                f"runtime_commit={runtime_commit or 'unknown'} "
                f"feature_schema_hash={self.predictor.metadata.get('feature_schema_hash', 'unknown')}"
            )
            if artifact_commit and runtime_commit and artifact_commit != runtime_commit:
                bt.logging.warning(
                    "Model artifact commit does not match current checkout | "
                    f"artifact_commit={artifact_commit} runtime_commit={runtime_commit}"
                )
        whitelist = sorted(self.validator_hotkey_whitelist)
        bt.logging.info(
            "Access policy | "
            f"force_validator_permit={self.config.blacklist.force_validator_permit} "
            f"allow_non_registered={self.config.blacklist.allow_non_registered} "
            f"validator_allowlist_count={len(whitelist)}"
        )
        if whitelist:
            bt.logging.info(f"Validator allowlist={whitelist}")
        bt.logging.info(
            "Miner docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    @staticmethod
    def _caller_hotkey(synapse: DetectionSynapse) -> str:
        return getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _align_score_count(scores: list, n: int) -> list:
        """Force exactly one score per chunk.

        The live validator (subnet >=0.1.30) DISCARDS the entire response and
        scores the cycle 0 if len(scores) != len(chunks). A truncated/extra list
        (e.g. a partial predictor failure) would zero a whole eval cycle, so we
        always return exactly ``n`` scores: extras are dropped, shortfalls are
        padded with 0.0 (treated as human — the safe, non-false-positive value).
        """
        scores = list(scores)
        if len(scores) > n:
            return scores[:n]
        if len(scores) < n:
            return scores + [0.0] * (n - len(scores))
        return scores

    def _compress_chunk(self, chunk: list[dict]) -> list[dict]:
        limit = self.max_hands_per_chunk_eval
        if limit <= 0 or len(chunk) <= limit:
            return chunk
        if limit == 1:
            return [chunk[len(chunk) // 2]]

        last_index = len(chunk) - 1
        slots = limit - 1
        indices = {
            min(last_index, round(index * last_index / slots))
            for index in range(limit)
        }
        return [chunk[index] for index in sorted(indices)]

    @classmethod
    def _score_hand(cls, hand: dict) -> tuple[float, dict[str, float]]:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter((action.get("action_type") or "").lower() for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )
        aggressive_actions = action_counts.get("bet", 0) + action_counts.get("raise", 0)
        passive_actions = action_counts.get("call", 0) + action_counts.get("check", 0)

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        bet_ratio = action_counts.get("bet", 0) / meaningful_actions
        aggression_ratio = aggressive_actions / max(aggressive_actions + passive_actions, 1)
        street_depth = len(streets) / 4.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0
        player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
        action_diversity = len(
            [kind for kind in ("call", "check", "bet", "raise", "fold") if action_counts.get(kind, 0)]
        ) / 5.0

        score = 0.0
        score += 0.24 * cls._clamp01(street_depth)
        score += 0.16 * cls._clamp01(showdown_flag)
        score += 0.18 * cls._clamp01(call_ratio / 0.32)
        score += 0.10 * cls._clamp01(check_ratio / 0.28)
        score += 0.08 * cls._clamp01(player_count_signal)
        score += 0.10 * cls._clamp01(action_diversity / 0.60)
        score -= 0.14 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.12 * cls._clamp01(raise_ratio / 0.22)
        score -= 0.06 * cls._clamp01(bet_ratio / 0.18)
        score -= 0.08 * cls._clamp01(aggression_ratio / 0.55)

        features = {
            "call_ratio": call_ratio,
            "check_ratio": check_ratio,
            "fold_ratio": fold_ratio,
            "raise_ratio": raise_ratio,
            "bet_ratio": bet_ratio,
            "aggression_ratio": aggression_ratio,
            "street_depth": street_depth,
            "showdown_flag": showdown_flag,
        }
        return cls._clamp01(score), features

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores: list[float] = []
        call_ratios: list[float] = []
        aggression_ratios: list[float] = []
        street_depths: list[float] = []
        showdown_flags: list[float] = []

        for hand in chunk:
            hand_score, features = cls._score_hand(hand)
            hand_scores.append(hand_score)
            call_ratios.append(features["call_ratio"])
            aggression_ratios.append(features["aggression_ratio"])
            street_depths.append(features["street_depth"])
            showdown_flags.append(features["showdown_flag"])

        avg_score = sum(hand_scores) / len(hand_scores)
        consistency_bonus = 0.0
        if len(hand_scores) > 1:
            call_spread = max(call_ratios) - min(call_ratios)
            aggression_spread = max(aggression_ratios) - min(aggression_ratios)
            street_spread = max(street_depths) - min(street_depths)
            showdown_rate = sum(showdown_flags) / len(showdown_flags)

            consistency_bonus += 0.10 * cls._clamp01(1.0 - call_spread / 0.60)
            consistency_bonus += 0.08 * cls._clamp01(1.0 - aggression_spread / 0.70)
            consistency_bonus += 0.05 * cls._clamp01(1.0 - street_spread)
            consistency_bonus += 0.05 * cls._clamp01(showdown_rate / 0.60)

        return round(cls._clamp01(avg_score + consistency_bonus), 6)

    def _apply_batch_rank(self, scores: List[float]) -> List[float]:
        """True-rank per-batch remap of one validator batch's scores.

        Maps each chunk to its within-batch rank in [0,1] (argsort-of-argsort),
        then into a band so exactly the top ``target_fraction`` cross 0.5 every
        batch. Rank-based -> preserves the model's ordering (AP and recall@FPR
        are unchanged) and is immune to score scale (works identically across
        models). Per-batch stable, unlike a fixed absolute threshold.
        """
        n = len(scores)
        if n <= 1:
            return scores
        # stable true rank in [0,1]: sort indices by (score, index), position=rank
        order = sorted(range(n), key=lambda i: (scores[i], i))
        rank = [0.0] * n
        for pos, i in enumerate(order):
            rank[i] = pos / (n - 1)
        frac = self.batch_rank_target_fraction
        span = self.batch_rank_span
        lo = 0.5 - (1.0 - frac) * span  # -> output(rank=1-frac) == 0.5
        return [self._clamp01(lo + r * span) for r in rank]

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        caller = self._caller_hotkey(synapse)
        chunks = [self._compress_chunk(list(chunk or [])) for chunk in (synapse.chunks or [])]
        chunk_sizes = [len(chunk) for chunk in chunks]
        bt.logging.info(
            "Validator query received | "
            f"caller={caller} "
            f"incoming_chunk_count={len(chunks)} "
            f"chunk_size_range={ [min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else [0, 0] }"
        )
        if self.query_log_preview:
            bt.logging.info(
                "Validator query preview | "
                f"caller={caller} "
                f"first_chunk_hand_count={chunk_sizes[0] if chunk_sizes else 0}"
            )

        started = time.perf_counter()
        backend_used = self.backend
        component_debug = {}
        if self.predictor is not None:
            try:
                # Offload the CPU/GPU-bound inference to a worker thread so the
                # asyncio event loop stays free to verify/accept other validators'
                # requests. Running it inline blocks the loop for the whole batch,
                # which ages incoming nonces past the verification window
                # (NotVerifiedException: "Nonce is too old") under large snapshots.
                scores = await asyncio.to_thread(
                    self.predictor.predict_chunk_scores, chunks
                )
            except Exception as err:
                bt.logging.warning(
                    f"Predictor failure during chunk scoring: {err}. "
                    "Falling back to heuristic backend."
                )
                backend_used = f"{self.detector.PROFILE}-fallback"
                scores = self.detector.score_chunks(chunks)
            # Diagnostics only: a failure here must NOT clobber the production
            # scores computed above, so it gets its own guarded block.
            if (
                backend_used != "heuristic-fallback"
                and self.component_debug_logging
                and hasattr(self.predictor, "debug_score_components")
            ):
                try:
                    component_debug = await asyncio.to_thread(
                        self.predictor.debug_score_components, chunks
                    )
                except Exception as err:
                    bt.logging.warning(
                        f"debug_score_components failed (diagnostic only, "
                        f"scores kept): {err}"
                    )
        else:
            scores = self.detector.score_chunks(chunks)
        # Live validator discards the whole response (reward 0) on a count mismatch.
        scores = self._align_score_count(scores, len(chunks))
        scores = [self._clamp01(score) for score in scores]
        # Batch-rank is a calibration safety net for the heuristic backend only;
        # a trained artifact carries its own score_remap calibration.
        heuristic_mode = self.predictor is None or str(backend_used).endswith("fallback")
        if self.batch_rank_enabled and heuristic_mode and len(scores) >= 4:
            scores = self._apply_batch_rank(scores)
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)

        # Live-query capture (input-only, env-gated POKER44_CAPTURE=1). A live
        # query has no ground-truth label; used only for benchmark->live OOD
        # diagnosis. Fire-and-forget in a worker thread so JSON serialization /
        # file IO never blocks the event loop or adds to response latency.
        if live_capture is not None and (live_capture.enabled() or live_capture.batch_enabled()):
            try:
                def _do_capture(ch, sc, uid, v):
                    if live_capture.enabled():
                        live_capture.capture(ch, sc, uid, v)
                    if live_capture.batch_enabled():
                        live_capture.capture_batch(ch, sc, uid, v)
                _task = asyncio.create_task(
                    asyncio.to_thread(
                        _do_capture, chunks, scores, getattr(self, "uid", "self"), caller
                    )
                )
                _CAPTURE_TASKS.add(_task)
                _task.add_done_callback(_CAPTURE_TASKS.discard)
            except Exception:
                pass

        bot_count = sum(1 for prediction in synapse.predictions if prediction)
        human_count = len(scores) - bot_count
        score_log_decimals = 8

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_hands = sum(chunk_sizes)
        per_chunk_ms = elapsed_ms / max(len(chunks), 1)
        per_hand_ms = elapsed_ms / max(total_hands, 1)
        score_range = (
            [
                round(min(scores), score_log_decimals),
                round(max(scores), score_log_decimals),
            ]
            if scores
            else [0.0, 0.0]
        )
        message = (
            f"Scored {len(chunks)} chunks with backend={backend_used} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"per_chunk_ms={per_chunk_ms:.2f} "
            f"per_hand_ms={per_hand_ms:.2f} "
            f"chunk_size_range={ [min(chunk_sizes), max(chunk_sizes)] if chunk_sizes else [0, 0] } "
            f"bot_count={bot_count} human_count={human_count} "
            f"score_range={score_range}"
        )
        if self.query_log_preview:
            message += (
                f" score_preview={scores[:5]} "
                f"prediction_preview={synapse.predictions[:5]}"
            )
        if component_debug:
            for name, values in component_debug.items():
                if not values:
                    continue
                message += (
                    f" {name}_range="
                    f"{[round(min(values), score_log_decimals), round(max(values), score_log_decimals)]}"
                )
        bt.logging.info(message)
        if self.score_array_logging:
            score_payload = {
                "chunk_sizes": chunk_sizes,
                "bot_count": bot_count,
                "human_count": human_count,
                "risk_scores": [
                    round(float(score), score_log_decimals) for score in scores
                ],
                "predictions": [bool(prediction) for prediction in synapse.predictions],
            }
            if component_debug:
                score_payload["components"] = {
                    name: [round(float(value), score_log_decimals) for value in values]
                    for name, values in component_debug.items()
                }
            bt.logging.info(f"Detailed chunk scores | {score_payload}")
        bt.logging.success(
            "Validator response sent successfully | "
            f"caller={caller} "
            f"incoming_chunk_count={len(chunks)} "
            f"risk_scores_length={len(scores)} "
            f"elapsed_ms={elapsed_ms:.2f} "
            f"per_chunk_ms={per_chunk_ms:.2f} "
            f"per_hand_ms={per_hand_ms:.2f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        blocked, reason = self.common_blacklist(synapse)
        caller = self._caller_hotkey(synapse)
        if blocked:
            bt.logging.warning(
                f"Blocked miner request | caller={caller} reason={reason}"
            )
        else:
            bt.logging.info(
                f"Accepted miner request | caller={caller} reason={reason}"
            )
        return blocked, reason

    async def priority(self, synapse: DetectionSynapse) -> float:
        caller = self._caller_hotkey(synapse)
        priority = self.caller_priority(synapse)
        bt.logging.debug(
            f"Assigned caller priority | caller={caller} priority={priority}"
        )
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
