from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np

from config import MAX_FRAMES, CANDIDATE_FPS, EMBEDDING_BATCH_SIZE, EARLY_STOP_MIN_DIST
from frame_sampler import seek_and_read
from frame_embedder import embed_frames

logger = logging.getLogger(__name__)


@dataclass
class SelectedFrame:
    frame_index: int
    timestamp_sec: float
    scene_id: int
    novelty_score: float
    image: np.ndarray


def _candidate_frame_numbers(scene, candidate_fps: float, video_fps: float) -> list:
    """Frame numbers to consider inside a scene.

    Samples ~candidate_fps across [start_frame, end_frame); always forces the
    scene start and end frames into the pool as guaranteed candidates.
    """
    start = scene.start_frame
    end = scene.end_frame - 1  # end_frame is exclusive, so last real frame is end-1
    if end < start:
        end = start

    step = max(1, int(round(video_fps / candidate_fps)))
    nums = list(range(start, end + 1, step))

    # Force scene boundaries in regardless of sampling step.
    if start not in nums:
        nums.append(start)
    if end not in nums:
        nums.append(end)

    return sorted(set(nums))


def _farthest_point_selection(embeddings: np.ndarray, budget: int, min_dist_threshold: float = 0.0) -> list:
    """Greedy max-min (farthest-point) selection over normalized embeddings.

    Works in euclidean space; with unit-norm embeddings this is equivalent to
    maximizing the minimum cosine distance. Seeds with the first candidate for
    determinism. Returns selected indices into the candidate array.

    If `min_dist_threshold > 0`, selection stops early once the next best
    max-min distance falls below the threshold (remaining candidates are
    near-duplicates of already-selected frames). This is ADDITIVE to `budget`:
    the MAX_FRAMES cap still bounds the count from above, this only stops sooner.
    """
    n = len(embeddings)
    if budget >= n:
        logger.info(f"[select] budget {budget} >= candidates {n} -> keeping ALL candidates")
        return list(range(n))

    selected = [0]
    min_dist = np.linalg.norm(embeddings - embeddings[0], axis=1)
    logger.info(f"[select] seed candidate idx=0 (t={0:.2f}s)")
    while len(selected) < budget:
        nxt = int(np.argmax(min_dist))
        best_dist = float(min_dist[nxt])
        if min_dist_threshold > 0.0 and best_dist < min_dist_threshold and len(selected) >= 1:
            logger.info(
                f"[select] early-stop: next max-min dist={best_dist:.4f} < "
                f"threshold={min_dist_threshold:.4f} -> keeping {len(selected)} diverse frame(s)"
            )
            break
        selected.append(nxt)
        new_dist = np.linalg.norm(embeddings - embeddings[nxt], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        logger.info(
            f"[select] picked idx={nxt} | max-min dist={best_dist:.4f} | "
            f"selected={len(selected)}/{budget}"
        )
    return selected


def select_keyframes(video_path: str, scenes: list) -> list:
    """Select the smallest good set of representative keyframes for a video.

    Single clean entry point so a later orchestrator (captioning/styling/Docker)
    can call this without knowing the internals. Candidates are pooled across the
    WHOLE video before embedding + selection, so both a single-scene clip and a
    20-scene clip bottom out at the same MAX_FRAMES ceiling.

    Returns SelectedFrame objects (temporally ordered) with frame_index,
    timestamp_sec, scene_id, novelty_score, and the BGR image itself.
    """
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"[selector] video_fps={video_fps:.3f} | CANDIDATE_FPS={CANDIDATE_FPS} | MAX_FRAMES={MAX_FRAMES}")

    # 1. Build the global candidate pool (scene start/end always included).
    candidates = []
    per_scene_counts = []
    for scene in scenes:
        cands = _candidate_frame_numbers(scene, CANDIDATE_FPS, video_fps)
        per_scene_counts.append(len(cands))
        for fn in cands:
            candidates.append({
                "frame_index": fn,
                "timestamp_sec": fn / video_fps,
                "scene_id": scene.scene_number,
            })
    # Dedupe by frame index while preserving temporal order.
    seen = set()
    unique = []
    for c in candidates:
        if c["frame_index"] not in seen:
            seen.add(c["frame_index"])
            unique.append(c)
    candidates = unique
    logger.info(
        f"[selector] candidates: raw={sum(per_scene_counts)} "
        f"({per_scene_counts} per scene) -> deduped={len(candidates)} "
        f"across {len(scenes)} scene(s)"
    )

    # 2. Read all candidate frames (one open capture, reused seek logic).
    images = []
    valid = []
    for c in candidates:
        frame = seek_and_read(cap, c["frame_index"])
        if frame is not None:
            images.append(frame)
            valid.append(c)
    cap.release()
    skipped = len(candidates) - len(images)
    logger.info(f"[selector] read {len(images)} candidate frames ({skipped} seeks failed)")

    if not images:
        logger.warning("[selector] no candidate frames read -> returning empty")
        return []

    # 3. Embed the entire pooled candidate set in one batched pass.
    t_emb = time.time()
    embeddings = embed_frames(images, EMBEDDING_BATCH_SIZE)
    logger.info(
        f"✅ [selector] embedded {embeddings.shape[0]} frames -> dim {embeddings.shape[1]} "
        f"in {time.time() - t_emb:.2f}s"
    )

    # 4. Novelty score: 1 - cosine_sim(embed[t], embed[t-1]) in temporal order.
    #    Diagnostic/debug only -> still written into output metadata.
    novelty = [0.0] * len(embeddings)
    for t in range(1, len(embeddings)):
        novelty[t] = 1.0 - float(np.dot(embeddings[t], embeddings[t - 1]))
    nov_arr = np.array(novelty[1:])
    logger.info(
        f"[selector] novelty(1-cos): mean={nov_arr.mean():.3f} "
        f"min={nov_arr.min():.3f} max={nov_arr.max():.3f}"
    )

    # 5. Greedy farthest-point selection, hard-capped at MAX_FRAMES globally.
    budget = min(MAX_FRAMES, len(embeddings))
    logger.info(f"[selector] farthest-point budget = min(MAX_FRAMES={MAX_FRAMES}, candidates={len(embeddings)}) = {budget}")
    selected_idx = _farthest_point_selection(embeddings, budget)

    # Assemble result in temporal order.
    result = []
    for idx in sorted(selected_idx):
        c = valid[idx]
        result.append(SelectedFrame(
            frame_index=c["frame_index"],
            timestamp_sec=c["timestamp_sec"],
            scene_id=c["scene_id"],
            novelty_score=novelty[idx],
            image=images[idx],
        ))

    logger.info(
        f"✅ [selector] DONE: {len(result)} keyframes selected in {time.time() - t0:.2f}s "
        f"(global ceiling respected: {len(result)} <= {MAX_FRAMES})"
    )
    return result
