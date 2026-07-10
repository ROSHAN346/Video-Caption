from dataclasses import dataclass
import logging

from scenedetect import SceneManager, AdaptiveDetector
from scenedetect.video_stream import VideoStream
from scenedetect.frame_timecode import FrameTimecode
from config import DETECTOR_CONFIG
from services.proxy_stream import ProxyStream

logger = logging.getLogger(__name__)

class MemoryVideoStream(VideoStream):
    def __init__(self, proxy: ProxyStream):
        self.proxy = proxy
        self.pos = 0
        self._base = FrameTimecode(timecode=0, fps=self.proxy.fps)

    @property
    def frame_rate(self): return self.proxy.fps
    
    @property
    def base_timecode(self): return self._base

    @property
    def position(self): return FrameTimecode(timecode=self.pos, fps=self.proxy.fps)
    
    @property
    def position_ms(self): return (self.pos / self.proxy.fps) * 1000.0
    
    @property
    def frame_number(self): return self.pos
    
    @property
    def aspect_ratio(self): return 1.0
    
    @property
    def duration(self): return FrameTimecode(timecode=len(self.proxy.frames), fps=self.proxy.fps)
    
    @property
    def is_seekable(self): return True
    
    @property
    def frame_size(self): return (self.proxy.width, self.proxy.height)
    
    @property
    def name(self): return "memory_stream"
    
    @property
    def path(self): return self.proxy.video_path

    def read(self, decode=True):
        if self.pos < len(self.proxy.frames):
            frame = self.proxy.frames[self.pos]
            self.pos += 1
            return frame if decode else True
        return False

    def reset(self): 
        self.pos = 0

    def seek(self, target):
        if isinstance(target, FrameTimecode):
            self.pos = target.frame_num
        elif isinstance(target, int):
            self.pos = target
        elif isinstance(target, float):
            self.pos = int(target * self.proxy.fps)



@dataclass
class Scene:
    scene_number: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


def detect_scenes(proxy: ProxyStream) -> list[Scene]:
    logger.info(f"[scenes] detector config: {DETECTOR_CONFIG}")
    video = MemoryVideoStream(proxy)
    scene_manager = SceneManager()
    scene_manager.auto_downscale = False
    scene_manager.downscale = 1
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
