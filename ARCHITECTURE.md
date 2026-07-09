# System Architecture — Video Keyframe Extraction & AI Report Generation

## 1. Executive Summary

This document describes the complete system architecture for a video analysis pipeline that extracts keyframes, analyzes them using Hugging Face Vision models, and generates multi-tone reports using Large Language Models. The architecture is modular, extensible, and designed for both CLI and web UI usage.

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                              │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │   CLI (main.py)      │              │   Web UI (app.py)    │            │
│  │   - argparse         │              │   - Streamlit        │            │
│  │   - sys.argv         │              │   - Session state    │            │
│  └──────────┬───────────┘              └──────────┬───────────┘            │
│             │                                      │                        │
└─────────────┼──────────────────────────────────────┼────────────────────────┘
              │                                      │
              ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORE PIPELINE LAYER                                 │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Scene     │    │   Frame     │    │   Frame     │    │   Frame     │  │
│  │  Detector   │───▶│  Sampler    │───▶│  Embedder   │───▶│  Selector   │  │
│  │             │    │             │    │  (CLIP)     │    │  (Farthest  │  │
│  │ AdaptiveDet │    │  Middle/    │    │  ViT-B/32   │    │   Point)    │  │
│  │             │    │  First/Last │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  Input: Video File          Output: Keyframes + scenes.json + keyframes.json│
└─────────────────────────────────────────────────────────────────────────────┘
              │
              │ (keyframes.jpg + scenes.json)
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI ANALYSIS LAYER (NEW)                                │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   HF Client     │    │   Image         │    │   Scene         │         │
│  │   (huggingface  │◀──▶│   Analyzer      │───▶│   Aggregator    │         │
│  │    _hub)        │    │                 │    │                 │         │
│  │                 │    │   Vision Model  │    │   Merge frame   │         │
│  │   InferenceClient│   │   Qwen2.5-VL    │    │   analyses      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Prompt        │    │   Report        │    │   Report        │         │
│  │   Loader        │───▶│   Generator     │───▶│   Cache         │         │
│  │                 │    │                 │    │                 │         │
│  │   Templates     │    │   Text Model   │    │   File-based    │         │
│  │   (5 styles)    │    │   Qwen3-8B      │    │   caching       │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │   PDF           │                                                       │
│  │   Generator     │                                                       │
│  │                 │                                                       │
│  │   fpdf2         │                                                       │
│  └─────────────────┘                                                       │
│                                                                             │
│  Output: analysis/*.json + reports/*/*.md + reports/*/*.pdf                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Core Pipeline Components

#### 3.1.1 Scene Detector (`scene_detector.py`)

```
┌────────────────────────────────────────────┐
│            SceneDetector                    │
├────────────────────────────────────────────┤
│ - Input: video_path (str)                  │
│ - Output: List[Scene]                      │
│                                            │
│ Dependencies:                              │
│   - scenedetect (AdaptiveDetector)         │
│   - config.py (DETECTOR_CONFIG)            │
│                                            │
│ Data Flow:                                 │
│   video_path → open_video()                │
│             → SceneManager.detect_scenes() │
│             → get_scene_list()             │
│             → Scene dataclass list         │
└────────────────────────────────────────────┘

Scene Dataclass:
┌────────────────────────────────────────────┐
│ @dataclass                                 │
│ class Scene:                               │
│   scene_number: int                        │
│   start_frame: int                         │
│   end_frame: int                           │
│   start_time: float                        │
│   end_time: float                          │
│   duration: float                          │
└────────────────────────────────────────────┘
```

#### 3.1.2 Frame Sampler (`frame_sampler.py`)

```
┌────────────────────────────────────────────┐
│            FrameSampler                    │
├────────────────────────────────────────────┤
│ - Input: Scene, video_path                 │
│ - Output: frame_number (int)               │
│                                            │
│ Strategies:                                │
│   - MiddleFrameSampler                     │
│   - FirstFrameSampler                      │
│   - LastFrameSampler                       │
│                                            │
│ Utility:                                   │
│   - seek_and_read(cap, frame_num)          │
│     → cv2.VideoCapture seek + read         │
└────────────────────────────────────────────┘
```

#### 3.1.3 Frame Embedder (`frame_embedder.py`)

```
┌────────────────────────────────────────────┐
│            FrameEmbedder                   │
├────────────────────────────────────────────┤
│ - Input: List[numpy.ndarray] (BGR frames)  │
│ - Output: numpy.ndarray (N, 512)           │
│                                            │
│ Dependencies:                              │
│   - clip (OpenAI CLIP)                     │
│   - torch                                  │
│   - PIL                                    │
│                                            │
│ Model:                                     │
│   - ViT-B/32 (default)                     │
│   - Device: CPU                            │
│   - Output: 512-dim L2-normalized vectors  │
│                                            │
│ Process:                                   │
│   1. BGR → RGB conversion                  │
│   2. CLIP preprocessing                    │
│   3. Batch inference (32 frames/batch)     │
│   4. L2 normalization                      │
│   5. Concatenate embeddings                │
└────────────────────────────────────────────┘
```

#### 3.1.4 Frame Selector (`frame_selector.py`)

```
┌────────────────────────────────────────────┐
│            FrameSelector                   │
├────────────────────────────────────────────┤
│ - Input: video_path, scenes                │
│ - Output: List[SelectedFrame]              │
│                                            │
│ Algorithm: Greedy Farthest-Point Selection │
│                                            │
│ Steps:                                     │
│   1. Build global candidate pool           │
│      - CANDIDATE_FPS sampling per scene    │
│      - Scene boundaries always included    │
│      - Deduplicate by frame index          │
│                                            │
│   2. Read all candidate frames             │
│      - Single VideoCapture opened          │
│      - seek_and_read() for each candidate  │
│                                            │
│   3. Embed all candidates                  │
│      - Batch embedding via CLIP            │
│      - L2-normalized vectors               │
│                                            │
│   4. Calculate novelty scores              │
│      - 1 - cosine_sim(t, t-1)             │
│                                            │
│   5. Farthest-point selection              │
│      - Seed with first frame               │
│      - Iteratively add farthest point      │
│      - Stop at MAX_FRAMES or early-stop    │
│                                            │
│ SelectedFrame Dataclass:                   │
│   - frame_index: int                       │
│   - timestamp_sec: float                   │
│   - scene_id: int                          │
│   - novelty_score: float                   │
│   - image: numpy.ndarray                   │
└────────────────────────────────────────────┘
```

---

### 3.2 AI Analysis Components (New)

#### 3.2.1 HF Client (`services/hf_client.py`)

```
┌────────────────────────────────────────────────────────────┐
│                      HFClient                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 InferenceClient                      │  │
│  │                 (huggingface_hub)                    │  │
│  │                                                      │  │
│  │  - api_token: str                                    │  │
│  │  - provider: str (nebius/together/fireworks)         │  │
│  │  - model: str                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ analyze_image(image_path, prompt, model) → str       │  │
│  │   - Read image file                                  │  │
│  │   - Base64 encode                                    │  │
│  │   - Build chat_completion message                    │  │
│  │   - Send to vision model                             │  │
│  │   - Return text response                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ generate_text(prompt, model, system_prompt) → str    │  │
│  │   - Build message list                               │  │
│  │   - Optional system prompt                           │  │
│  │   - Send to text model                               │  │
│  │   - Return generated text                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ analyze_image_base64(image_bytes, prompt, model)     │  │
│  │   - Accept raw bytes                                 │  │
│  │   - Base64 encode in memory                          │  │
│  │   - Send to vision model                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.2 Image Analyzer (`services/image_analyzer.py`)

```
┌────────────────────────────────────────────────────────────┐
│                   ImageAnalyzer                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: Keyframe JPEG path                                │
│  Output: Structured JSON (Scene Analysis)                 │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Analysis Prompt Template                │  │
│  │                                                      │  │
│  │  "Analyze this image and return a JSON object with   │  │
│  │   the following fields:                              │  │
│  │   - scene_type: (indoor/outdoor/mixed/abstract)      │  │
│  │   - location: (office/street/nature/room/etc.)       │  │
│  │   - people: (count and description if visible)       │  │
│  │   - objects: (list of main objects)                  │  │
│  │   - vehicles: (list if any)                          │  │
│  │   - animals: (list if any)                           │  │
│  │   - activities: (what is happening)                  │  │
│  │   - weather: (sunny/cloudy/rainy/etc.)              │  │
│  │   - time_of_day: (morning/afternoon/evening/night)   │  │
│  │   - environment: (urban/rural/industrial/natural)    │  │
│  │   - risk_level: (low/medium/high with reason)        │  │
│  │   - confidence: (0.0-1.0)                           │  │
│  │   - summary: (1-2 sentence description)             │  │
│  │   Return ONLY valid JSON."                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Output Schema:                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {                                                    │  │
│  │   "scene_id": 1,                                     │  │
│  │   "scene_type": "outdoor",                           │  │
│  │   "location": "city street",                         │  │
│  │   "people": "2 pedestrians walking",                 │  │
│  │   "objects": ["car", "traffic light", "building"],   │  │
│  │   "vehicles": ["white sedan"],                       │  │
│  │   "animals": [],                                     │  │
│  │   "activities": "cars driving, people walking",      │  │
│  │   "weather": "sunny",                                │  │
│  │   "time_of_day": "afternoon",                        │  │
│  │   "environment": "urban",                            │  │
│  │   "risk_level": "low - normal traffic",              │  │
│  │   "confidence": 0.92,                                │  │
│  │   "summary": "A busy city street with light traffic" │  │
│  │ }                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - analyze_keyframe(image_path, hf_client) → dict         │
│  - analyze_keyframes_batch(paths, hf_client) → list[dict] │
│  - validate_json(response) → dict                          │
│  - retry_analysis(image_path, hf_client, attempts=3)      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.3 Scene Aggregator (`services/scene_aggregator.py`)

```
┌────────────────────────────────────────────────────────────┐
│                  SceneAggregator                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: List[dict] (per-frame analyses)                   │
│  Output: dict (aggregated scene analysis)                 │
│                                                            │
│  Aggregation Strategy:                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                      │  │
│  │  Frame 1 Analysis ──┐                                │  │
│  │  Frame 2 Analysis ──┼──▶ Consensus ──▶ Scene JSON   │  │
│  │  Frame 3 Analysis ──┘                                │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Field Merging Rules:                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ scene_type    → majority vote                        │  │
│  │ location      → majority vote                        │  │
│  │ people        → concatenate unique descriptions      │  │
│  │ objects       → union of all lists (deduplicated)    │  │
│  │ vehicles      → union of all lists                   │  │
│  │ animals       → union of all lists                   │  │
│  │ activities    → merge and deduplicate                 │  │
│  │ weather       → majority vote                        │  │
│  │ time_of_day   → majority vote                        │  │
│  │ environment   → majority vote                        │  │
│  │ risk_level    → max severity wins                    │  │
│  │ confidence    → average of all                       │  │
│  │ summary       → combine into single paragraph        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - aggregate_scene_analyses(analyses) → dict              │
│  - consensus_field(values) → str                          │
│  - merge_lists(lists) → list                              │
│  - merge_people_descriptions(descriptions) → str          │
│  - merge_activities(activities) → str                     │
│  - merge_risk_levels(risks) → str                         │
│  - calculate_overall_confidence(analyses) → float         │
│  - merge_summaries(summaries) → str                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.4 Prompt Loader (`services/prompt_loader.py`)

```
┌────────────────────────────────────────────────────────────┐
│                   PromptLoader                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Directory: prompts/                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ prompts/                                             │  │
│  │ ├── formal.txt                                       │  │
│  │ ├── sarcastic.txt                                    │  │
│  │ ├── humorous_tech.txt                                │  │
│  │ ├── humorous_non_tech.txt                            │  │
│  │ └── jargon.txt                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Template Format:                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {style_instructions}                                 │  │
│  │                                                      │  │
│  │ Scene Data:                                          │  │
│  │ {scene_data}                                         │  │
│  │                                                      │  │
│  │ Write a {style} report based on this scene analysis. │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - load_prompt(style) → str                               │
│  - format_prompt(template, scene_data) → str              │
│  - get_available_styles() → list[str]                     │
│  - validate_style(style) → bool                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.5 Report Generator (`services/report_generator.py`)

```
┌────────────────────────────────────────────────────────────┐
│                  ReportGenerator                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: scene_data (dict), style (str)                    │
│  Output: report_text (str)                                │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Generation Flow                      │  │
│  │                                                      │  │
│  │  scene_data ──▶ prompt_loader.load_prompt(style)     │  │
│  │              ──▶ prompt_loader.format_prompt()       │  │
│  │              ──▶ hf_client.generate_text()           │  │
│  │              ──▶ report_cache.cache_report()         │  │
│  │              ──▶ return report_text                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  System Prompts per Style:                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ formal     → "You are a professional report writer." │  │
│  │ sarcastic  → "You are a witty, sarcastic commentator"│  │
│  │ humorous_tech → "You are a programmer who finds..."  │  │
│  │ humorous_non_tech → "You are a stand-up comedian."   │  │
│  │ jargon     → "You are a domain expert in jargon."    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - generate_report(scene_data, style, hf_client) → str   │
│  - generate_all_reports(scene_data, hf_client) → dict    │
│  - get_system_prompt(style) → str                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.6 Report Cache (`services/report_cache.py`)

```
┌────────────────────────────────────────────────────────────┐
│                    ReportCache                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Storage Structure:                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ reports/                                             │  │
│  │ └── {video_name}/                                    │  │
│  │     └── scene_{scene_id}/                            │  │
│  │         ├── formal.md                                │  │
│  │         ├── formal.pdf                               │  │
│  │         ├── sarcastic.md                             │  │
│  │         ├── sarcastic.pdf                            │  │
│  │         ├── humorous_tech.md                         │  │
│  │         ├── humorous_tech.pdf                        │  │
│  │         ├── humorous_non_tech.md                     │  │
│  │         ├── humorous_non_tech.pdf                    │  │
│  │         ├── jargon.md                                │  │
│  │         ├── jargon.pdf                               │  │
│  │         └── cache.json                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  cache.json Format:                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {                                                    │  │
│  │   "scene_id": 1,                                     │  │
│  │   "generated_at": "2026-07-08T22:30:00Z",            │  │
│  │   "vision_model": "Qwen/Qwen2.5-VL-7B-Instruct",    │  │
│  │   "text_model": "Qwen/Qwen3-8B-Instruct",           │  │
│  │   "styles": ["formal", "sarcastic", ...],            │  │
│  │   "analysis_hash": "abc123..."                       │  │
│  │ }                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - get_cache_path(scene_id, style) → Path                 │
│  - get_cached_report(scene_id, style) → str|None          │
│  - cache_report(scene_id, style, report, metadata)        │
│  - is_cached(scene_id, style) → bool                      │
│  - clear_cache(scene_id=None)                             │
│  - get_cache_stats() → dict                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### 3.2.7 PDF Generator (`services/pdf_generator.py`)

```
┌────────────────────────────────────────────────────────────┐
│                   PDFGenerator                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: report_text (str), keyframe_path (str)            │
│  Output: PDF file                                         │
│                                                            │
│  PDF Layout:                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ┌────────────────────────────────────────────────┐   │  │
│  │ │              KEYFRAME IMAGE                    │   │  │
│  │ │              (header)                          │   │  │
│  │ └────────────────────────────────────────────────┘   │  │
│  │                                                      │  │
│  │ Scene Analysis Report                                │  │
│  │ ────────────────────────                             │  │
│  │ Scene ID: 1                                          │  │
│  │ Style: Formal                                        │  │
│  │ Generated: 2026-07-08 22:30:00                       │  │
│  │                                                      │  │
│  │ ┌────────────────────────────────────────────────┐   │  │
│  │ │  Scene Metadata Table                          │   │  │
│  │ │  ─────────────────────                         │   │  │
│  │ │  Scene Type: outdoor                           │   │  │
│  │ │  Location: city street                         │   │  │
│  │ │  Risk Level: low                               │   │  │
│  │ │  Confidence: 0.92                              │   │  │
│  │ └────────────────────────────────────────────────┘   │  │
│  │                                                      │  │
│  │ Report Content:                                      │  │
│  │ ─────────────────                                    │  │
│  │ {generated_report_text}                              │  │
│  │                                                      │  │
│  │ ──────────────────────────────────────────────────   │  │
│  │ Generated by Video Keyframe Extractor v1.0           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Methods:                                                  │
│  - markdown_to_pdf(md_text, output_path) → Path           │
│  - generate_report_pdf(scene_id, style, report,           │
│                        keyframe_path, scene_data) → Path  │
│  - add_header_page(pdf, keyframe_path, scene_data)        │
│  - add_metadata_table(pdf, scene_data)                    │
│  - add_report_content(pdf, report_text)                   │
│  - add_footer(pdf)                                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Architecture

### 4.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: VIDEO PROCESSING (Existing)                                      │
│  ══════════════════════════════════════                                     │
│                                                                             │
│  [video.mp4]                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │ Scene       │ ──▶ [scenes.json]                                         │
│  │ Detection   │      (scene boundaries)                                    │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Keyframe    │ ──▶ [keyframe_000.jpg, keyframe_001.jpg, ...]            │
│  │ Selection   │      [keyframes.json]                                      │
│  └──────┬──────┘      (selected frames + metadata)                         │
│         │                                                                   │
│  PHASE 2: AI ANALYSIS (New)                                                │
│  ════════════════════════════                                               │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Image       │                                                            │
│  │ Analyzer    │ ──▶ [analysis/scene_001_frame_000.json]                   │
│  │ (HF Vision) │      [analysis/scene_001_frame_001.json]                  │
│  └──────┬──────┘      (per-frame structured analysis)                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Scene       │                                                            │
│  │ Aggregator  │ ──▶ [analysis/scene_001_aggregated.json]                  │
│  └──────┬──────┘      (merged scene analysis)                              │
│         │                                                                   │
│  PHASE 3: REPORT GENERATION (New)                                          │
│  ═══════════════════════════════════                                        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Prompt      │                                                            │
│  │ Loader      │ ──▶ Loads template from prompts/{style}.txt               │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Report      │                                                            │
│  │ Generator   │ ──▶ [reports/scene_001/formal.md]                         │
│  │ (HF Text)   │      [reports/scene_001/sarcastic.md]                     │
│  └──────┬──────┘      [reports/scene_001/humorous_tech.md]                 │
│         │             [reports/scene_001/humorous_non_tech.md]              │
│         │             [reports/scene_001/jargon.md]                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ PDF         │                                                            │
│  │ Generator   │ ──▶ [reports/scene_001/formal.pdf]                        │
│  └─────────────┘      [reports/scene_001/sarcastic.pdf]                     │
│                        [reports/scene_001/humorous_tech.pdf]                │
│                        [reports/scene_001/humorous_non_tech.pdf]            │
│                        [reports/scene_001/jargon.pdf]                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 API Call Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API CALL SEQUENCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Image Analysis (per keyframe):                                             │
│  ─────────────────────────────                                              │
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │ Read     │────▶│ Base64   │────▶│ Build    │────▶│ Send to  │          │
│  │ JPEG     │     │ Encode   │     │ Message  │     │ HF API   │          │
│  └──────────┘     └──────────┘     └──────────┘     └────┬─────┘          │
│                                                           │                │
│                              ┌────────────────────────────┘                │
│                              ▼                                             │
│                    ┌──────────────────┐                                    │
│                    │ Qwen2.5-VL-7B    │                                    │
│                    │ (Vision Model)   │                                    │
│                    └────────┬─────────┘                                    │
│                             │                                              │
│                             ▼                                              │
│                    ┌──────────────────┐                                    │
│                    │ JSON Response    │                                    │
│                    │ (scene analysis) │                                    │
│                    └──────────────────┘                                    │
│                                                                             │
│  Report Generation (per scene per style):                                   │
│  ────────────────────────────────────────                                   │
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │ Load     │────▶│ Format   │────▶│ Build    │────▶│ Send to  │          │
│  │ Template │     │ with     │     │ Message  │     │ HF API   │          │
│  └──────────┘     │ scene    │     └──────────┘     └────┬─────┘          │
│                   │ data     │                            │                │
│                   └──────────┘                            ▼                │
│                                            ┌──────────────────┐            │
│                                            │ Qwen3-8B         │            │
│                                            │ (Text Model)     │            │
│                                            └────────┬─────────┘            │
│                                                     │                      │
│                                                     ▼                      │
│                                            ┌──────────────────┐            │
│                                            │ Markdown Report  │            │
│                                            └──────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Configuration Architecture

### 5.1 Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Priority (highest to lowest):                                              │
│                                                                             │
│  1. Environment Variables (.env)                                            │
│     └── HF_API_TOKEN=hf_xxxxx                                              │
│                                                                             │
│  2. CLI Arguments (main.py)                                                 │
│     └── --reports, --style formal                                           │
│                                                                             │
│  3. Streamlit Sidebar (app.py)                                              │
│     └── Model selectors, style checkboxes                                   │
│                                                                             │
│  4. Config File (config.py)                                                 │
│     └── All default values                                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │ config.py                                            │                   │
│  │                                                      │                   │
│  │ # Existing Config                                    │                   │
│  │ DETECTOR_CONFIG = { ... }                            │                   │
│  │ FRAME_STRATEGY = "middle"                            │                   │
│  │ MAX_FRAMES = 200                                     │                   │
│  │ CANDIDATE_FPS = 5.0                                  │                   │
│  │ CLIP_MODEL_NAME = "ViT-B/32"                         │                   │
│  │ EMBEDDING_BATCH_SIZE = 32                            │                   │
│  │ EARLY_STOP_MIN_DIST = 0.03                           │                   │
│  │                                                      │                   │
│  │ # NEW: Hugging Face Config                           │                   │
│  │ HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")        │                   │
│  │ HF_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"    │                   │
│  │ HF_TEXT_MODEL = "Qwen/Qwen3-8B-Instruct"            │                   │
│  │ HF_VISION_PROVIDER = "nebius"                        │                   │
│  │ HF_TEXT_PROVIDER = "nebius"                          │                   │
│  │                                                      │                   │
│  │ # NEW: Report Config                                 │                   │
│  │ REPORT_STYLES = [                                    │                   │
│  │     "formal", "sarcastic", "humorous_tech",          │                   │
│  │     "humorous_non_tech", "jargon"                    │                   │
│  │ ]                                                    │                   │
│  │ DEFAULT_REPORT_STYLE = "formal"                      │                   │
│  │ REPORT_CACHE_ENABLED = True                          │                   │
│  └──────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Error Handling Architecture

### 6.1 Error Types and Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ERROR HANDLING STRATEGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┬───────────────────────────────────────────────┐    │
│  │ Error Type          │ Handling Strategy                             │    │
│  ├─────────────────────┼───────────────────────────────────────────────┤    │
│  │ API Token Invalid   │ Clear error message + setup instructions     │    │
│  │ API Rate Limit      │ Exponential backoff (1s, 2s, 4s, 8s)        │    │
│  │ Model Unavailable   │ Fallback to alternative model                │    │
│  │ Invalid JSON Output │ Retry with simpler prompt (max 3 attempts)   │    │
│  │ Network Timeout     │ Retry with increased timeout                 │    │
│  │ File Not Found      │ Skip and log warning                        │    │
│  │ Disk Space          │ Check before writing, warn if low            │    │
│  └─────────────────────┴───────────────────────────────────────────────┘    │
│                                                                             │
│  Retry Logic:                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ def retry_with_backoff(func, max_attempts=3, base_delay=1.0):      │   │
│  │     for attempt in range(max_attempts):                             │   │
│  │         try:                                                        │   │
│  │             return func()                                           │   │
│  │         except RateLimitError:                                      │   │
│  │             delay = base_delay * (2 ** attempt)                     │   │
│  │             time.sleep(delay)                                       │   │
│  │         except ModelUnavailableError:                               │   │
│  │             return fallback_model_call()                            │   │
│  │     raise MaxRetriesExceeded()                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Security Architecture

### 7.1 API Token Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY MEASURES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  API Token Storage:                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ .env file (not committed to git)                                    │   │
│  │ ─────────────────────────────────                                   │   │
│  │ HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx                           │   │
│  │                                                                      │   │
│  │ .gitignore must include:                                            │   │
│  │ - .env                                                              │   │
│  │ - *.key                                                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Token Usage:                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Never logged or printed                                           │   │
│  │ - Never stored in config.py (only in .env)                          │   │
│  │ - Loaded via python-dotenv                                          │   │
│  │ - Passed to InferenceClient constructor                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Input Validation:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Validate image files before processing                            │   │
│  │ - Sanitize JSON responses before parsing                            │   │
│  │ - Check file paths for directory traversal                          │   │
│  │ - Limit batch sizes to prevent memory exhaustion                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Architecture

### 8.1 Optimization Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE OPTIMIZATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Caching Strategy                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Report cache prevents regeneration                                │   │
│  │ - Analysis cache stores per-frame JSON                              │   │
│  │ - Cache invalidation: manual or TTL-based                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. Batch Processing                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Process multiple keyframes in sequence                            │   │
│  │ - Single HF client instance (connection pooling)                    │   │
│  │ - Configurable batch sizes                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. Lazy Loading                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - HF client initialized on first use                                │   │
│  │ - Prompts loaded on demand                                          │   │
│  │ - PDF generation only when requested                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Estimated Processing Times:                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Operation                    │ Time (estimate)                      │   │
│  │ ─────────────────────────────┼───────────────────────────────────── │   │
│  │ Image analysis (1 frame)     │ 2-5 seconds                         │   │
│  │ Image analysis (100 frames)  │ 3-8 minutes                         │   │
│  │ Report generation (1 style)  │ 3-8 seconds                         │   │
│  │ Report generation (5 styles) │ 15-40 seconds                       │   │
│  │ PDF generation (1 report)    │ 1-2 seconds                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Scalability Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCALABILITY NOTES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Current Design (Single Machine):                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Sequential processing per video                                   │   │
│  │ - File-based caching                                                │   │
│  │ - In-memory state (Streamlit session)                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Future Scaling Options:                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Redis cache for distributed caching                               │   │
│  │ - Celery/Redis for async task queue                                 │   │
│  │ - PostgreSQL for metadata storage                                   │   │
│  │ - Docker for containerized deployment                               │   │
│  │ - Kubernetes for orchestration                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  API Rate Limit Considerations:                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Free tier: ~$0.10/month                                           │   │
│  │ - PRO tier: ~$2/month included                                      │   │
│  │ - Per-image: ~$0.001                                                │   │
│  │ - Per-report: ~$0.002                                               │   │
│  │ - 100 keyframes + 500 reports ≈ $1.10                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Testing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TESTING STRATEGY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Unit Tests:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - test_hf_client.py: Mock API calls                                 │   │
│  │ - test_image_analyzer.py: Test JSON validation                      │   │
│  │ - test_scene_aggregator.py: Test merging logic                      │   │
│  │ - test_prompt_loader.py: Test template loading                      │   │
│  │ - test_report_generator.py: Test report creation                    │   │
│  │ - test_report_cache.py: Test cache operations                       │   │
│  │ - test_pdf_generator.py: Test PDF creation                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Integration Tests:                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - test_full_pipeline.py: End-to-end video → reports                 │   │
│  │ - test_api_integration.py: Real API calls (with token)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Mock Strategy:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Use pytest-mock for API mocking                                   │   │
│  │ - Record real API responses for replay                              │   │
│  │ - Test error scenarios with mocked failures                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Option 1: Local Development                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Run directly with Python                                          │   │
│  │ - Use .env for API token                                            │   │
│  │ - Streamlit for UI                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Option 2: Docker                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Dockerfile:                                                         │   │
│  │ FROM python:3.12-slim                                               │   │
│  │ WORKDIR /app                                                        │   │
│  │ COPY requirements.txt .                                             │   │
│  │ RUN pip install -r requirements.txt                                 │   │
│  │ COPY . .                                                            │   │
│  │ EXPOSE 8501                                                         │   │
│  │ CMD ["streamlit", "run", "app.py"]                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Option 3: Cloud Deployment                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ - Streamlit Cloud (free tier)                                       │   │
│  │ - AWS EC2 / GCP Compute                                             │   │
│  │ - Docker + Kubernetes                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. File Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FILE DEPENDENCY GRAPH                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  config.py                                                                  │
│      │                                                                      │
│      ├──▶ scene_detector.py                                                │
│      │        │                                                             │
│      │        ▼                                                             │
│      │    frame_sampler.py                                                  │
│      │        │                                                             │
│      │        ▼                                                             │
│      │    frame_embedder.py ◀── (CLIP model)                               │
│      │        │                                                             │
│      │        ▼                                                             │
│      │    frame_selector.py                                                 │
│      │        │                                                             │
│      ▼        ▼                                                             │
│    main.py ──────────────────────────────────────────────────────────┐     │
│    app.py                                                            │     │
│                                                                      │     │
│  ┌───────────────────────────────────────────────────────────────────┘     │
│  │                                                                         │
│  ▼                                                                         │
│  services/                                                                 │
│  ├── hf_client.py ◀──────── (huggingface_hub.InferenceClient)            │
│  │       │                                                                 │
│  │       ├──▶ image_analyzer.py                                           │
│  │       │        │                                                        │
│  │       │        ▼                                                        │
│  │       │    scene_aggregator.py                                          │
│  │       │        │                                                        │
│  │       │        ▼                                                        │
│  │       │    prompt_loader.py ◀── prompts/*.txt                          │
│  │       │        │                                                        │
│  │       │        ▼                                                        │
│  │       │    report_generator.py                                          │
│  │       │        │                                                        │
│  │       │        ├──▶ report_cache.py                                     │
│  │       │        │                                                         │
│  │       │        ▼                                                        │
│  │       │    pdf_generator.py                                             │
│  │       │                                                                 │
│  └───────┴─────────────────────────────────────────────────────────────────┘
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Core | Python | 3.12+ | Main language |
| Video | OpenCV | 5.0+ | Video I/O, image processing |
| Scene Detection | PySceneDetect | 0.7+ | Adaptive scene detection |
| Embeddings | OpenCLIP | latest | CLIP ViT-B/32 embeddings |
| Deep Learning | PyTorch | 2.0+ | Neural network inference |
| HF API | huggingface_hub | 0.25+ | InferenceClient for APIs |
| PDF | fpdf2 | 2.8+ | PDF generation |
| Config | python-dotenv | 1.0+ | Environment variable loading |
| Web UI | Streamlit | 1.28+ | Interactive web interface |
| Vision Model | Qwen2.5-VL-7B | - | Image analysis |
| Text Model | Qwen3-8B | - | Report generation |

---

## Appendix B: Model Comparison

### Vision Models

| Model | Parameters | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| Qwen2.5-VL-7B | 7B | Medium | High | ~$0.001/image |
| SmolVLM | 2B | Fast | Medium | ~$0.0005/image |
| Florence-2 | 0.7B | Very Fast | Medium | ~$0.0003/image |

### Text Models

| Model | Parameters | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| Qwen3-8B | 8B | Medium | High | ~$0.002/report |
| Llama-3.1-8B | 8B | Medium | High | ~$0.002/report |
| Mistral Nemo | 12B | Slow | High | ~$0.003/report |

---

*Document Version: 1.0*
*Last Updated: 2026-07-08*
*Author: Video AMD Project Team*
