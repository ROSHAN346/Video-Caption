from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np

from config import MAX_FRAMES, CANDIDATE_FPS, EMBEDDING_BATCH_SIZE, EARLY_STOP_MIN_DIST
from frame_embedder import embed_frames
from services.proxy_stream import ProxyStream

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
    """
    n = len(embeddings)
    if n == 0:
        return []
        
    if budget >= n and min_dist_threshold <= 0.0:
        logger.info(f"[select] budget {budget} >= candidates {n} -> keeping ALL candidates")
        return list(range(n))

    budget = min(budget, n)
    selected = [0]
    if n == 1:
        return selected
        
    min_dist = np.linalg.norm(embeddings - embeddings[0], axis=1)
    logger.info(f"[select] seed candidate idx=0")
    
    while len(selected) < budget:
        nxt = int(np.argmax(min_dist))
        best_euclidean = float(min_dist[nxt])
        
        # Convert Euclidean distance of unit-norm vectors back to Cosine distance
        # Euclidean = sqrt(2 * (1 - cos_sim)) -> Cosine Dist = 1 - cos_sim = (Euclidean^2) / 2
        best_cosine = 0.5 * (best_euclidean ** 2)
        
        if min_dist_threshold > 0.0 and best_cosine < min_dist_threshold and len(selected) >= 1:
            logger.info(
                f"[select] early-stop: next max-min cosine_dist={best_cosine:.4f} < "
                f"threshold={min_dist_threshold:.4f} -> keeping {len(selected)} diverse frame(s)"
            )
            break
            
        selected.append(nxt)
        new_dist = np.linalg.norm(embeddings - embeddings[nxt], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        logger.info(
            f"[select] picked idx={nxt} | max-min cosine_dist={best_cosine:.4f} | "
            f"selected={len(selected)}/{budget}"
        )
        
    return selected


def select_keyframes(proxy: ProxyStream, scenes: list) -> tuple[list, dict]:
    """Select the smallest good set of representative keyframes for a video.

    Returns:
        tuple of (SelectedFrame list, frame_stats dict)
    """
    t0 = time.time()
    video_fps = proxy.fps
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

    # 2. Read all candidate frames from the in-memory proxy stream.
    images = []
    valid = []
    for c in candidates:
        frame = proxy.get_frame(c["frame_index"])
        if frame is not None:
            images.append(frame)
            valid.append(c)
    skipped = len(candidates) - len(images)
    logger.info(f"[selector] read {len(images)} candidate frames from proxy ({skipped} out-of-bounds)")

    if not images:
        logger.warning("[selector] no candidate frames read -> returning empty")
        return [], {"candidates": len(candidates), "read": 0, "selected": 0, "pruned_read": skipped, "pruned_sim": len(candidates)}

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

    # 5. Greedy farthest-point selection, PER SCENE, adaptive budget.
    scene_to_indices = {}
    for idx, c in enumerate(valid):
        scene_to_indices.setdefault(c["scene_id"], []).append(idx)
        
    selected_idx = []
    
    for scene_id, indices in scene_to_indices.items():
        if not indices:
            continue
            
        # Distribute MAX_FRAMES proportional to candidate count (duration proxy)
        weight = len(indices) / len(valid)
        scene_budget = max(1, int(round(MAX_FRAMES * weight)))
        scene_budget = min(scene_budget, len(indices))
        
        scene_embeddings = embeddings[indices]
        logger.info(f"[selector] Scene {scene_id} budget={scene_budget} candidates={len(indices)}")
        
        # Apply min_dist_threshold to enable early stopping
        local_selected = _farthest_point_selection(
            scene_embeddings, 
            budget=scene_budget, 
            min_dist_threshold=EARLY_STOP_MIN_DIST
        )
        
        # Map local indices back to global indices
        for local_idx in local_selected:
            selected_idx.append(indices[local_idx])

    # Enforce global hard ceiling (trimming from the end is safe enough as a fallback)
    if len(selected_idx) > MAX_FRAMES:
        logger.warning(f"[selector] Trimming global excess: {len(selected_idx)} -> {MAX_FRAMES}")
        selected_idx = selected_idx[:MAX_FRAMES]

    # Assemble result in temporal order.
    result = []
    for idx in sorted(selected_idx):
        c = valid[idx]
        
        # Decode the full-resolution frame for the selected timestamp
        full_res_image = proxy.get_full_res_frame(c["frame_index"])
        if full_res_image is None:
            logger.warning(f"[selector] Failed to extract full-res for frame {c['frame_index']}. Using proxy.")
            full_res_image = images[idx]
            
        result.append(SelectedFrame(
            frame_index=c["frame_index"],
            timestamp_sec=c["timestamp_sec"],
            scene_id=c["scene_id"],
            novelty_score=novelty[idx],
            image=full_res_image,
        ))

    logger.info(
        f"[selector] done in {time.time()-t0:.1f}s. "
        f"Selected {len(result)}/{len(images)} valid frames."
    )
    
    frame_stats = {
        "candidates": len(candidates),
        "read": len(images),
        "selected": len(result),
        "pruned_read": skipped,
        "pruned_sim": len(images) - len(result)
    }

    return result, frame_stats
