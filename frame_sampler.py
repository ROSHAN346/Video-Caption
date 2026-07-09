from abc import ABC, abstractmethod
import cv2
from scene_detector import Scene
from config import FRAME_STRATEGY


def seek_and_read(cap: cv2.VideoCapture, frame_num: int):
    """Seek an already-opened capture to frame_num and return the BGR frame.

    Reused by frame_selector.py so the cv2 seek logic lives in exactly one place.
    Returns None if the seek/read fails.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    return frame if ret else None


class FrameSampler(ABC):
    @abstractmethod
    def sample(self, scene: Scene) -> int:
        """Return frame number to extract for given scene."""
        pass


class MiddleFrameSampler(FrameSampler):
    def sample(self, scene: Scene) -> int:
        return (scene.start_frame + scene.end_frame) // 2


class FirstFrameSampler(FrameSampler):
    def sample(self, scene: Scene) -> int:
        return scene.start_frame


class LastFrameSampler(FrameSampler):
    def sample(self, scene: Scene) -> int:
        return scene.end_frame - 1


SAMPLERS = {
    "middle": MiddleFrameSampler,
    "first": FirstFrameSampler,
    "last": LastFrameSampler,
}


def get_sampler(strategy: str) -> FrameSampler:
    if strategy not in SAMPLERS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[strategy]()


def extract_frames(video_path: str, scenes: list[Scene], output_dir: str, strategy: str = FRAME_STRATEGY) -> list[str]:
    sampler = get_sampler(strategy)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    filenames = []
    for scene in scenes:
        frame_num = sampler.sample(scene)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            filename = f"scene_{scene.scene_number:03d}.jpg"
            filepath = f"{output_dir}/{filename}"
            cv2.imwrite(filepath, frame)
            filenames.append(filename)

    cap.release()
    return filenames
