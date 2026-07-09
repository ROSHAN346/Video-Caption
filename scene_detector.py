from dataclasses import dataclass
import logging

from scenedetect import open_video, SceneManager, AdaptiveDetector
from config import DETECTOR_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    scene_number: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


def detect_scenes(video_path: str) -> list[Scene]:
    logger.info(f"[scenes] detector config: {DETECTOR_CONFIG}")
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(
        adaptive_threshold=DETECTOR_CONFIG["adaptive_threshold"],
        min_scene_len=DETECTOR_CONFIG["min_scene_len"],
        window_width=DETECTOR_CONFIG["window_width"],
        min_content_val=DETECTOR_CONFIG["min_content_val"],
    ))
    scene_manager.detect_scenes(video, show_progress=True)
    raw_scenes = scene_manager.get_scene_list(start_in_scene=True)
    logger.info(f"[scenes] raw scene cuts returned: {len(raw_scenes)}")

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

    logger.info(f"✅ [scenes] {len(scenes)} scene(s) detected:")
    for s in scenes:
        logger.info(
            f"  scene {s.scene_number}: frames {s.start_frame}-{s.end_frame} "
            f"t={s.start_time:.2f}-{s.end_time:.2f}s dur={s.duration:.2f}s"
        )
    return scenes
