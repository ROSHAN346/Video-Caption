# video-amd — AI Video Captioning Pipeline

Given a list of video URLs, the pipeline: downloads each video → detects scenes →
samples & CLIP-embeds candidate frames → selects up to 5 diverse keyframes →
analyzes them with a Fireworks vision model → aggregates per scene → generates
short captions in multiple writing styles via a Fireworks text model → writes
`results.json`.

## Project layout

```
video-amd/
├── main.py                # CLI entrypoints (batch/competition + single video)
├── app.py                 # Optional Streamlit UI wrapper
├── config.py              # All tunables + env-driven API config
├── scene_detector.py      # Scene detection (single decode pass, PySceneDetect)
├── frame_selector.py      # Keyframe selection (farthest-point over CLIP embeddings)
├── frame_embedder.py      # CLIP ViT-B/32 embeddings (CUDA / DirectML / CPU)
├── services/
│   ├── fireworks_client.py    # Vision API client (OpenAI-compatible)
│   │                          # (vision + text use separate Fireworks keys)
│   ├── image_analyzer.py      # Structured JSON frame analysis
│   ├── scene_aggregator.py    # Consensus aggregation per scene
│   ├── report_generator.py    # Styled caption generation
│   ├── report_cache.py        # File cache for single-video reports
│   └── prompt_loader.py       # Loads prompts/*.txt style templates
├── prompts/               # formal / sarcastic / humorous_tech / humorous_non_tech
├── input/tasks.json       # Sample batch input
├── Dockerfile, docker-entrypoint.sh
└── requirements*.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in FIREWORKS_API_KEY (vision) and FIREWORKS_TEXT_API_KEY (text)
```

## Usage

**Batch / competition mode:**

```bash
python main.py --input input/tasks.json --output out/results.json
```

Input format:

```json
[
  {
    "task_id": "my_clip",
    "video_url": "https://.../clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

Output: `[{"task_id": "...", "captions": {"<style>": "<caption>"}}]`.
Unknown styles are mapped to the nearest supported style.

**Single video mode:**

```bash
python main.py video.mp4 --reports --style formal      # one style
python main.py video.mp4 --reports --all-styles        # all styles
```

Writes keyframes, `keyframes.json`, `scenes.json`, per-frame analysis JSON, and
styled reports under `output/<video>/`.

**Web UI:**

```bash
pip install streamlit
streamlit run app.py
```

## Docker

```bash
# CPU (default)
docker build -t video-amd .
# AMD ROCm
docker build --build-arg TORCH_FLAVOR=rocm -t video-amd .

docker run --rm \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  video-amd
```

The container looks for `/input/tasks.json`, `/input/input.json`, or any
`*.json` mounted in `/input`, and writes to `$OUTPUT_PATH`
(default `/output/results.json`). Never bake API keys into the image; pass
them at runtime with `-e` or `--env-file`.

## Pipeline internals

1. **Scene detection** (`scene_detector.py`) — one decode pass; PySceneDetect
   `AdaptiveDetector` runs on downscaled frames while candidate frames (uniform
   grid at `CANDIDATE_FPS` + scene boundaries) are captured in memory, downscaled
   to `MAX_CAPTURE_SIDE` to bound RAM.
2. **Keyframe selection** (`frame_selector.py`) — pooled candidates are CLIP-embedded
   in batches, then greedy farthest-point selection picks up to `MAX_FRAMES`
   diverse frames; selection stops early when the next candidate is a
   near-duplicate (`EARLY_STOP_MIN_DIST`).
3. **Vision analysis** (`services/image_analyzer.py`) — each keyframe is JPEG-encoded
   in memory and sent to the Fireworks vision model, which returns structured JSON
   (scene type, objects, activities, risk level, summary, ...). Runs in a thread
   pool with retry/backoff on rate limits.
4. **Aggregation** (`services/scene_aggregator.py`) — per-scene consensus over frame
   analyses (majority vote on scalars, merged lists, max risk).
5. **Caption generation** (`services/report_generator.py`) — scene data is injected
   into a style template and sent to the Fireworks text model (key #2); one
   caption per style,
   generated in parallel, with local template fallbacks on failure.

Key tunables live in `config.py` (`MAX_FRAMES`, `CANDIDATE_FPS`,
`EARLY_STOP_MIN_DIST`, `EMBEDDING_BATCH_SIZE`, `DETECTOR_CONFIG`,
`MAX_DOWNLOAD_MB`, `MAX_CAPTURE_SIDE`).
