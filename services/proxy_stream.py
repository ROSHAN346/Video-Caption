import av
import numpy as np
import logging


logger = logging.getLogger(__name__)

class ProxyStream:
    """
    Decodes the video once into a low-resolution in-memory proxy stream
    for both Scene Detection and Candidate Selection.
    """
    def __init__(self, video_path: str, proxy_side: int = 360):
        self.video_path = video_path
        self.proxy_side = proxy_side
        self.frames = []
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self._load()

    def _load(self):
        logger.info(f"Decoding video {self.video_path} into in-memory PyAV proxy stream (max_side={self.proxy_side})...")
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        
        # Handle division by zero for fps
        if stream.average_rate:
            self.fps = float(stream.average_rate)
            
        w = stream.codec_context.width
        h = stream.codec_context.height
        
        long = max(w, h)
        if long > self.proxy_side:
            scale = self.proxy_side / float(long)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = w, h
            
        self.width = new_w
        self.height = new_h
            
        for frame in container.decode(stream):
            # Fast downscale using libswscale inside PyAV
            img = frame.reformat(width=new_w, height=new_h, format='bgr24').to_ndarray()
            self.frames.append(img)
            
        container.close()
        logger.info(f"Proxy stream loaded {len(self.frames)} frames")

    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a proxy frame by index."""
        if 0 <= frame_num < len(self.frames):
            return self.frames[frame_num]
        return None

    def get_full_res_frame(self, frame_num: int) -> np.ndarray:
        """Decode exactly one full-resolution frame using PyAV."""
        logger.info(f"Extracting full-resolution frame {frame_num} using PyAV...")
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        
        # Seek accurately
        # PyAV seeks by PTS. We calculate target PTS.
        pts_per_frame = int(stream.duration / stream.frames) if stream.frames and stream.duration else None
        
        # If we can't reliably seek, or it's inaccurate, we just seek a bit before and decode
        # OpenCV seek was used before, let's just use PyAV seek
        target_pts = int((frame_num / self.fps) / stream.time_base)
        container.seek(target_pts, stream=stream, backward=True)
        
        # decode until we hit the exact frame
        result = None
        for frame in container.decode(stream):
            # approximate frame index based on time
            curr_frame = int(frame.time * self.fps)
            if curr_frame >= frame_num:
                result = frame.to_ndarray(format='bgr24')
                break
                
        container.close()
        
        # Fallback if PyAV seek fails
        if result is None:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, result = cap.read()
            cap.release()
            
        return result
