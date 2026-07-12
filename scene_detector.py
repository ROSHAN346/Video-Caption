from dataclasses import dataclass
import logging
from collections import deque

import cv2
import numpy as np
from scenedetect import SceneManager, AdaptiveDetector, FrameTimecode
from scenedetect.scene_manager import compute_downscale_factor

from config import DETECTOR_CONFIG, CANDIDATE_FPS, MAX_CAPTURE_SIDE

logger = logging.getLogger(__name__)


def _shrink_for_capture(frame: np.ndarray) -> np.ndarray:
    """Downscale a frame so its longest side is <= MAX_CAPTURE_SIDE.

    Captured frames are held in memory for the whole video; on UHD sources this
    would otherwise consume many GB. CLIP embedding (224px) and vision-API JPEGs
    (1280px cap) never need more resolution than this.
    """
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_CAPTURE_SIDE:
        return frame
    scale = MAX_CAPTURE_SIDE / float(long_side)
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


@dataclass
class Scene:
    scene_number: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


def detect_scenes(video_path: str):
    """Detect scenes AND capture candidate keyframe images in a single decode pass.

    Returns a tuple (scenes, captured_frames) where:
      - scenes: list[Scene]
      - captured_frames: dict[frame_index -> BGR ndarray] containing a uniform
        temporal grid (spaced ~1/CANDIDATE_FPS) plus every detected scene
        boundary frame. These feed straight into select_keyframes() so the video
        is decoded exactly once instead of being re-decoded for keyframe reading.

    Detection reuses PySceneDetect's AdaptiveDetector (driven per-frame via
    SceneManager._process_frame) while frames are captured at full resolution.
    Detection runs on a downscaled copy (matching PySceneDetect's internal
    behaviour) so it stays fast; only the captured pixels stay full-res.
    """
    logger.info(f"[scenes] detector config: {DETECTOR_CONFIG} | CANDIDATE_FPS={CANDIDATE_FPS}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Downscale factor used purely for detection speed (mirrors PySceneDetect).
    if w > 0 and h > 0:
        factor = compute_downscale_factor(max(w, h))
        dw, dh = max(1, int(round(w / factor))), max(1, int(round(h / factor)))
    else:
        factor, dw, dh = 1, w, h

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(
        adaptive_threshold=DETECTOR_CONFIG["adaptive_threshold"],
        min_scene_len=DETECTOR_CONFIG["min_scene_len"],
        window_width=DETECTOR_CONFIG["window_width"],
        min_content_val=DETECTOR_CONFIG["min_content_val"],
    ))
    # We drive _process_frame ourselves, so disable SceneManager's own downscale.
    scene_manager.auto_downscale = False
    scene_manager._base_timecode = FrameTimecode(0, fps)
    scene_manager._start_pos = FrameTimecode(0, fps)

    step = max(1, int(round(fps / CANDIDATE_FPS)))
    captured: dict[int, np.ndarray] = {}
    recent: deque[tuple[int, np.ndarray]] = deque(maxlen=32)
    frame_idx = 0
    last_cuts = 0

    logger.info("Detecting scenes (single decode pass)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tc = FrameTimecode(frame_idx, fps)
        if factor > 1:
            small = cv2.resize(frame, (dw, dh))
        else:
            small = frame
        scene_manager._process_frame(tc, small)

        # Capture pixels: uniform temporal grid + scene boundaries. Frames stored
        # in the long-lived `captured` dict are downscaled to bound memory; the
        # `recent` deque holds at most 32 frames so it stays as-is (no resize cost).
        if frame_idx % step == 0:
            captured[frame_idx] = _shrink_for_capture(frame)
        recent.append((frame_idx, frame))

        new_cuts = scene_manager._cutting_list[last_cuts:]
        for cut_tc in new_cuts:
            ci = cut_tc.frame_num
            for ri, rf in recent:
                if ri == ci and ci not in captured:
                    captured[ci] = _shrink_for_capture(rf)
                if ri == ci - 1 and (ci - 1) not in captured:
                    captured[ci - 1] = _shrink_for_capture(rf)
        if new_cuts:
            last_cuts = len(scene_manager._cutting_list)

        frame_idx += 1

    # Ensure the final frame is captured (end of the last scene).
    if frame_idx > 0 and (frame_idx - 1) not in captured:
        captured[frame_idx - 1] = _shrink_for_capture(recent[-1][1])

    cap.release()
    scene_manager._last_pos = FrameTimecode(frame_idx, fps)
    scene_manager._post_process(FrameTimecode(frame_idx, fps))
    raw_scenes = scene_manager.get_scene_list(start_in_scene=True)
    logger.info(f"[scenes] raw scene cuts returned: {len(raw_scenes)} | captured {len(captured)} frames")

    scenes = []
    for i, (start_tc, end_tc) in enumerate(raw_scenes, 1):
        scenes.append(Scene(
            scene_number=i,
            start_frame=start_tc.frame_num,
            end_frame=end_tc.frame_num,
            start_time=start_tc.seconds,
            end_time=end_tc.seconds,
            duration=round(end_tc.seconds - start_tc.seconds, 3),
        ))

    logger.info(f"[scenes] {len(scenes)} scene(s) detected:")
    for s in scenes:
        logger.info(
            f"  scene {s.scene_number}: frames {s.start_frame}-{s.end_frame} "
            f"t={s.start_time:.2f}-{s.end_time:.2f}s dur={s.duration:.2f}s"
        )
    return scenes, captured
