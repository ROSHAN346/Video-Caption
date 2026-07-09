# Video Keyframe Extraction Pipeline — Project Documentation

## 1. Project Overview

This project is an automated video keyframe extraction pipeline that identifies and extracts the most representative frames from a video. It combines **scene detection** (finding shot boundaries) with **CLIP-based semantic embeddings** (understanding visual content) to select diverse, meaningful keyframes.

### Goals

- Detect scene boundaries in videos automatically
- Sample candidate frames densely within each scene
- Embed frames using CLIP for semantic understanding
- Select the most diverse keyframes using farthest-point selection
- Output JPEG keyframes with JSON metadata

### Key Technologies

| Technology | Purpose |
|------------|---------|
| PySceneDetect | Adaptive scene boundary detection |
| OpenCLIP (CLIP) | Visual embedding for semantic similarity |
| PyTorch | Neural network inference |
| OpenCV | Video I/O and image processing |
| Streamlit | Web-based user interface |

---

## 2. Architecture & File Structure

```
video-amd-main/
├── config.py              # Central configuration parameters
├── scene_detector.py      # Scene detection using AdaptiveDetector
├── frame_sampler.py       # Frame sampling strategies (middle/first/last)
├── frame_embedder.py      # CLIP-based frame embedding
├── frame_selector.py      # Keyframe selection using farthest-point algorithm
├── main.py                # CLI entry point
├── app.py                 # Streamlit web UI
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── video1.mp4             # Sample video (test input)
├── video2.mp4             # Sample video (test input)
└── output/                # Generated output directory
    └── <video_name>/
        ├── keyframe_000.jpg
        ├── keyframe_001.jpg
        ├── ...
        ├── keyframes.json
        └── scenes.json
```

### Module Responsibilities

| File | Lines | Responsibility |
|------|-------|----------------|
| `config.py` | 27 | All tunable parameters in one place |
| `scene_detector.py` | 51 | Detect scene boundaries, return `Scene` dataclass list |
| `frame_sampler.py` | 71 | Extract single frames per scene (legacy/utility) |
| `frame_embedder.py` | 49 | Load CLIP model, embed BGR frames to normalized vectors |
| `frame_selector.py` | 186 | Orchestrate candidate sampling, embedding, and selection |
| `main.py` | 129 | CLI entry point, runs full pipeline, saves outputs |
| `app.py` | 281 | Streamlit web interface with configuration sliders |

---

## 3. Dependencies

```txt
opencv-python>=4.5.0      # Video capture, image I/O, resize
numpy>=1.20.0             # Array operations, embedding math
torch>=1.9.0              # PyTorch for CLIP inference
Pillow>=8.0.0             # Image format conversion (PIL)
scenedetect>=0.5.0        # Scene boundary detection
git+https://github.com/openai/CLIP.git  # CLIP model
streamlit>=1.28.0         # Web UI framework
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Pipeline Walkthrough (Step-by-Step)

### Step 1: Configuration (`config.py`)

All pipeline parameters are centralized in `config.py`:

```python
# Scene Detection
DETECTOR_CONFIG = {
    "adaptive_threshold": 3.0,    # Ratio threshold for scene cuts
    "min_scene_len": 15,          # Minimum frames between cuts
    "window_width": 2,            # Rolling average window size
    "min_content_val": 15.0,      # Minimum content change to register
}

# Keyframe Selection
MAX_FRAMES = 200                 # Global cap on keyframes
CANDIDATE_FPS = 5.0              # Dense sampling rate within scenes
CLIP_MODEL_NAME = "ViT-B/32"    # CLIP model variant
EMBEDDING_BATCH_SIZE = 32        # Frames per forward pass
EARLY_STOP_MIN_DIST = 0.03       # Early stop threshold
FRAME_STRATEGY = "middle"        # Default frame sampling strategy
```

**Why this matters:** Centralized config ensures consistent behavior across CLI and web UI, and makes tuning easy.

---

### Step 2: Scene Detection (`scene_detector.py`)

**Input:** Video file path  
**Output:** List of `Scene` dataclass objects

```python
@dataclass
class Scene:
    scene_number: int      # 1-indexed scene ID
    start_frame: int       # First frame number
    end_frame: int         # Last frame number (exclusive)
    start_time: float      # Start time in seconds
    end_time: float        # End time in seconds
    duration: float        # Duration in seconds
```

#### How It Works

1. **Open Video:** Uses `scenedetect.open_video()` to get a video handle
2. **Configure Detector:** Creates `AdaptiveDetector` with parameters from `DETECTOR_CONFIG`
3. **Detect Scenes:** Runs two-pass detection:
   - **Pass 1:** Calculates per-frame `content_val` (HSV color differences)
   - **Pass 2:** Applies rolling average to compute `adaptive_ratio`
4. **Extract Boundaries:** Converts timecodes to `Scene` objects with frame numbers and timestamps

#### AdaptiveDetector Algorithm

The adaptive detector calculates a ratio for each frame:

```
adaptive_ratio = current_frame_score / average_surrounding_score
```

A scene cut is triggered when:
- `adaptive_ratio >= adaptive_threshold` (default 3.0)
- `content_val >= min_content_val` (default 15.0)
- `frames_since_last_cut >= min_scene_len` (default 15)

This approach handles camera motion better than fixed thresholds because it adjusts based on local context.

#### Code Flow

```python
# scene_detector.py:20-51
def detect_scenes(video_path: str) -> list[Scene]:
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
    # Convert to Scene dataclass list...
```

---

### Step 3: Candidate Frame Sampling (`frame_selector.py`)

**Input:** List of scenes, video FPS  
**Output:** Global pool of candidate frame numbers

#### Purpose

Instead of picking one frame per scene, the pipeline samples multiple candidates to find the most representative frames. This is controlled by `CANDIDATE_FPS` (default 5.0 frames/second).

#### Algorithm

```python
# frame_selector.py:24-44
def _candidate_frame_numbers(scene, candidate_fps: float, video_fps: float) -> list:
    start = scene.start_frame
    end = scene.end_frame - 1
    step = max(1, int(round(video_fps / candidate_fps)))
    nums = list(range(start, end + 1, step))

    # Force scene boundaries into the pool
    if start not in nums:
        nums.append(start)
    if end not in nums:
        nums.append(end)

    return sorted(set(nums))
```

#### Example

For a 30 FPS video with `CANDIDATE_FPS = 5.0`:
- `step = 30 / 5 = 6` frames between samples
- Scene frames 0-90 → samples at 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90
- Scene boundaries (0, 90) always included

#### Global Pool Assembly

All candidates from all scenes are pooled together:
- Duplicated frame indices are removed (preserving temporal order)
- Each candidate stores: `frame_index`, `timestamp_sec`, `scene_id`

---

### Step 4: CLIP Embedding (`frame_embedder.py`)

**Input:** List of BGR numpy frames  
**Output:** L2-normalized embedding matrix (N, D)

#### CLIP Model Loading

```python
# frame_embedder.py:16-22
def load_model():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load(CLIP_MODEL_NAME, device=_DEVICE)
        _model.eval()
    return _model, _preprocess
```

- Model is loaded once and cached (singleton pattern)
- Runs on CPU only (`_DEVICE = "cpu"`)
- Default model: `ViT-B/32` (lightweight, reasonable quality)

#### Embedding Process

```python
# frame_embedder.py:25-49
def embed_frames(frames: list, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    load_model()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensors = []
            for f in batch:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(_preprocess(Image.fromarray(rgb)))
            batch_t = torch.stack(tensors).to(_DEVICE)
            feats = _model.encode_image(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)
```

#### Key Details

1. **Color Conversion:** OpenCV uses BGR, CLIP expects RGB
2. **Preprocessing:** CLIP's built-in transform (resize, normalize, etc.)
3. **Batching:** Processes `EMBEDDING_BATCH_SIZE` frames at a time (default 32)
4. **Normalization:** L2-normalized to unit vectors (cosine similarity = dot product)
5. **No Gradients:** `torch.no_grad()` for inference efficiency

#### Output

- Shape: `(N, 512)` for ViT-B/32 (512-dimensional embeddings)
- All vectors have unit norm (||v|| = 1)

---

### Step 5: Novelty Scoring (`frame_selector.py`)

**Input:** Embedding matrix (N, D)  
**Output:** Novelty scores per frame

```python
# frame_selector.py:154-163
novelty = [0.0] * len(embeddings)
for t in range(1, len(embeddings)):
    novelty[t] = 1.0 - float(np.dot(embeddings[t], embeddings[t - 1]))
```

#### Formula

```
novelty[t] = 1 - cosine_similarity(embed[t], embed[t-1])
```

#### Interpretation

| Score | Meaning |
|-------|---------|
| 0.0 | Identical to previous frame |
| 0.5 | Moderately different |
| 1.0 | Completely different |

This score is stored in the output JSON for diagnostic purposes but does not drive the selection algorithm (farthest-point selection does).

---

### Step 6: Farthest-Point Keyframe Selection (`frame_selector.py`)

**Input:** Embedding matrix, budget (MAX_FRAMES)  
**Output:** List of selected indices

This is the core selection algorithm that chooses the most diverse keyframes.

#### Algorithm: Greedy Max-Min Farthest-Point Selection

```python
# frame_selector.py:47-83
def _farthest_point_selection(embeddings: np.ndarray, budget: int,
                               min_dist_threshold: float = 0.0) -> list:
    n = len(embeddings)
    if budget >= n:
        return list(range(n))  # Keep all if budget allows

    selected = [0]  # Seed with first frame
    min_dist = np.linalg.norm(embeddings - embeddings[0], axis=1)

    while len(selected) < budget:
        nxt = int(np.argmax(min_dist))  # Find farthest point
        best_dist = float(min_dist[nxt])

        # Early stop if next point is too close (near-duplicate)
        if min_dist_threshold > 0.0 and best_dist < min_dist_threshold:
            break

        selected.append(nxt)
        new_dist = np.linalg.norm(embeddings - embeddings[nxt], axis=1)
        min_dist = np.minimum(min_dist, new_dist)  # Update distances

    return selected
```

#### How It Works

1. **Seed:** Start with the first candidate frame (deterministic)
2. **Iterate:**
   - Find the candidate farthest from all currently selected frames
   - Add it to the selection
   - Update distance array (each point tracks its distance to nearest selected)
3. **Stop:** When budget is reached OR next point is too close (`EARLY_STOP_MIN_DIST`)

#### Why Farthest-Point?

- Maximizes visual diversity across keyframes
- Ensures coverage of entire video content
- Avoids redundant/near-duplicate frames
- Deterministic (same input → same output)

#### Budget Enforcement

```python
budget = min(MAX_FRAMES, len(embeddings))
```

The final selection never exceeds `MAX_FRAMES` (default 200).

---

### Step 7: Output Generation (`main.py`)

**Input:** Selected keyframes, scenes  
**Output:** JPEG files + JSON metadata

#### JPEG Saving with Downscaling

```python
# main.py:20-27
MAX_SAVE_SIDE = 1280

def _downscale_to_max_side(frame):
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= MAX_SAVE_SIDE:
        return frame
    scale = MAX_SAVE_SIDE / float(long)
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)
```

- UHD sources (4K) are downscaled to max 1280px on longest side
- Uses `INTER_AREA` interpolation (best for downscaling)
- Full-quality frames used for embedding/selection; only saved JPEGs are downscaled

#### keyframes.json Format

```json
[
  {
    "frame_index": 0,
    "timestamp_sec": 0.0,
    "scene_id": 1,
    "novelty_score": 0.0,
    "image_path": "keyframe_000.jpg"
  },
  {
    "frame_index": 45,
    "timestamp_sec": 1.5,
    "scene_id": 1,
    "novelty_score": 0.2341,
    "image_path": "keyframe_001.jpg"
  }
]
```

#### scenes.json Format

```json
[
  {
    "scene_number": 1,
    "start_time": "00:00:00.000",
    "end_time": "00:00:05.233",
    "duration": 5.233
  },
  {
    "scene_number": 2,
    "start_time": "00:00:05.233",
    "end_time": "00:00:12.100",
    "duration": 6.867
  }
]
```

---

## 5. Key Algorithms Explained

### 5.1 Adaptive Scene Detection

**Problem:** Fixed thresholds fail when video has varying motion levels (e.g., static shot followed by fast action).

**Solution:** Rolling average adjusts threshold based on local context.

```
For each frame:
  1. Calculate content_val (HSV color differences)
  2. Compute average content_val over [t-window, t+window]
  3. adaptive_ratio = content_val[t] / average
  4. If adaptive_ratio > threshold AND content_val > min_content_val:
       Mark scene cut
```

**Parameters:**
- `window_width=2`: Average over 2 frames before/after
- `adaptive_threshold=3.0`: Current frame must be 3x more different than local average
- `min_content_val=15.0`: Absolute minimum change to avoid noise

### 5.2 CLIP Visual Embedding

**What is CLIP?** A neural network trained on image-text pairs that creates semantically meaningful visual embeddings.

**Why CLIP?**
- Understands semantic content (not just pixel differences)
- Robust to lighting, angle, and minor variations
- Embeddings enable cosine similarity comparison

**Model:** ViT-B/32 (Vision Transformer, Base, 32x32 patches)
- Input: 224x224 RGB image
- Output: 512-dimensional vector
- Size: ~350MB (reasonable for CPU)

### 5.3 Greedy Farthest-Point Selection

**Problem:** Given N candidate frames, select K most diverse ones.

**Algorithm:**
```
selected = {first_frame}
distances = [distance_to_first_frame for each candidate]

while |selected| < K:
    candidate = argmax(distances)  # Farthest from any selected
    selected.add(candidate)
    distances = min(distances, distance_to_candidate)
```

**Complexity:** O(K × N) — efficient for typical video sizes

**Properties:**
- Maximizes minimum pairwise distance
- Provides coverage of entire visual space
- Deterministic and reproducible

---

## 6. CLI Usage

### Basic Usage

```bash
python main.py <video_path>
```

### Example

```bash
python main.py video1.mp4
```

### Output

```
12:34:56 | === START pipeline for: video1.mp4 ===
12:34:56 | [scenes] detector config: {'adaptive_threshold': 3.0, ...}
12:34:57 | ✅ [scenes] 5 scene(s) detected:
12:34:57 |   scene 1: frames 0-150 t=0.00-5.00s dur=5.00s
12:34:57 |   scene 2: frames 150-300 t=5.00-10.00s dur=5.00s
...
12:34:58 | [selector] candidates: raw=150 -> deduped=148 across 5 scene(s)
12:35:02 | ✅ [selector] embedded 148 frames -> dim 512 in 4.23s
12:35:03 | ✅ [selector] DONE: 25 keyframes selected in 7.45s
12:35:03 | Saved: keyframe_000.jpg | scene=1 frame=0 t=0.00s novelty=0.000
...
12:35:04 | ✅ === PHASE ARTIFACT COMPLETE ===
12:35:04 | ✅ VERIFY: scenes=5 | keyframes=25 (<=MAX_FRAMES=200) | total runtime=8.12s
```

### Output Directory

```
output/
└── video1/
    ├── keyframe_000.jpg
    ├── keyframe_001.jpg
    ├── ...
    ├── keyframe_024.jpg
    ├── keyframes.json
    └── scenes.json
```

---

## 7. Streamlit Web UI

### Launch

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Features

#### Sidebar Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Adaptive Threshold | 1.0 - 10.0 | 3.0 | Scene detection sensitivity |
| Min Scene Length | 5 - 100 frames | 15 | Minimum frames between cuts |
| Window Width | 1 - 10 | 2 | Rolling average window |
| Min Content Value | 5.0 - 50.0 | 15.0 | Absolute content threshold |
| Max Frames | 1 - 50 | 10 | Keyframe budget |
| Candidate FPS | 1.0 - 10.0 | 5.0 | Dense sampling rate |
| Early Stop Min Distance | 0.0 - 0.5 | 0.05 | Near-duplicate threshold |
| CLIP Model | ViT-B/32, ViT-B/16, ViT-L/14 | ViT-B/32 | Model variant |

#### Main Interface

1. **Upload Video:** Drag-and-drop or file picker (mp4, avi, mov, mkv, webm)
2. **Video Info:** Displays resolution, FPS, duration
3. **Extract Button:** Runs the full pipeline
4. **Progress Bar:** Shows detection → embedding → saving phases
5. **Results Grid:** Displays extracted keyframes with metadata
6. **Scene Table:** Breakdown of detected scenes
7. **Download Buttons:** Export keyframes.json and scenes.json

---

## 8. Configuration Reference

### Scene Detection Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `adaptive_threshold` | float | 3.0 | 1.0-10.0 | Ratio threshold for scene cuts |
| `min_scene_len` | int | 15 | 5-100 | Minimum frames between cuts |
| `window_width` | int | 2 | 1-10 | Rolling average window size |
| `min_content_val` | float | 15.0 | 5.0-50.0 | Minimum content change to register |

### Keyframe Selection Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `MAX_FRAMES` | int | 200 | 1-500 | Global cap on keyframes |
| `CANDIDATE_FPS` | float | 5.0 | 1.0-10.0 | Dense sampling rate within scenes |
| `CLIP_MODEL_NAME` | str | "ViT-B/32" | - | CLIP model variant |
| `EMBEDDING_BATCH_SIZE` | int | 32 | 8-128 | Frames per forward pass |
| `EARLY_STOP_MIN_DIST` | float | 0.03 | 0.0-0.5 | Early stop threshold |
| `FRAME_STRATEGY` | str | "middle" | - | Default sampling strategy |

### Tuning Guidelines

| Scenario | Adjust |
|----------|--------|
| Too many scenes detected | Increase `adaptive_threshold` |
| Missing scene cuts | Decrease `adaptive_threshold` |
| Too many keyframes | Decrease `MAX_FRAMES` |
| Missing important frames | Increase `CANDIDATE_FPS` |
| Slow processing | Decrease `CANDIDATE_FPS` or `MAX_FRAMES` |
| Near-duplicate keyframes | Increase `EARLY_STOP_MIN_DIST` |

---

## 9. Output Format

### Directory Structure

```
output/
└── <video_name>/
    ├── keyframe_000.jpg      # Extracted keyframe images
    ├── keyframe_001.jpg
    ├── ...
    ├── keyframes.json        # Keyframe metadata
    └── scenes.json           # Scene detection results
```

### keyframes.json Schema

```json
[
  {
    "frame_index": "int — Frame number in original video",
    "timestamp_sec": "float — Time in seconds",
    "scene_id": "int — Scene number (1-indexed)",
    "novelty_score": "float — 1 - cosine_sim with previous frame (0-1)",
    "image_path": "string — Filename of saved JPEG"
  }
]
```

### scenes.json Schema

```json
[
  {
    "scene_number": "int — Scene number (1-indexed)",
    "start_time": "string — HH:MM:SS.mmm format",
    "end_time": "string — HH:MM:SS.mmm format",
    "duration": "float — Duration in seconds"
  }
]
```

### Example Output

```json
// keyframes.json
[
  {"frame_index": 0, "timestamp_sec": 0.0, "scene_id": 1, "novelty_score": 0.0, "image_path": "keyframe_000.jpg"},
  {"frame_index": 45, "timestamp_sec": 1.5, "scene_id": 1, "novelty_score": 0.2341, "image_path": "keyframe_001.jpg"},
  {"frame_index": 90, "timestamp_sec": 3.0, "scene_id": 1, "novelty_score": 0.1876, "image_path": "keyframe_002.jpg"}
]

// scenes.json
[
  {"scene_number": 1, "start_time": "00:00:00.000", "end_time": "00:00:05.233", "duration": 5.233},
  {"scene_number": 2, "start_time": "00:00:05.233", "end_time": "00:00:12.100", "duration": 6.867}
]
```

---

## 10. Performance Notes

### Processing Time (Approximate)

| Video Length | Scenes | Candidates | Embedding Time | Total Time |
|--------------|--------|------------|----------------|------------|
| 30 seconds | 5 | ~150 | ~4s | ~8s |
| 2 minutes | 12 | ~600 | ~15s | ~25s |
| 10 minutes | 30 | ~3000 | ~60s | ~90s |

### Memory Usage

- CLIP ViT-B/32: ~350MB model
- Per frame embedding: ~2KB (512 floats)
- 1000 candidates: ~2MB embeddings

### Optimization Notes

- CLIP model loaded once per process (cached)
- Batching reduces GPU/CPU overhead
- Early stop reduces unnecessary selection iterations
- JPEG downscaling saves disk space without affecting selection quality
