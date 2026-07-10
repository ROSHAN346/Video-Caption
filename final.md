# Final Project Analysis Report — `video-amd-main`

> **Report generated:** 2026-07-10
> **Scope:** Full static analysis of every source file, configuration, documentation, and asset in the repository.
> **Method:** Manual source reading of all Python modules, service clients, prompts, docs, JSON I/O, and build files.

---

## 1. Executive Summary

`video-amd-main` is a **video keyframe-extraction + AI captioning pipeline** built for what appears to be an AMD hackathon (clips are hosted under `amd-hackathon-clips` and `tasks.json`/`results.json` follow a competition schema). It performs two independent phases:

1. **Phase A — Keyframe Extraction (offline, local):** Detect scene cuts with PySceneDetect, densely sample candidate frames, embed them with OpenAI **CLIP**, then select the most visually diverse keyframes via a **greedy farthest-point (max–min) selection**.
2. **Phase B — AI Captioning (online, API):** Send selected keyframes to a multimodal vision LLM, get structured JSON per frame, aggregate per scene, then generate **4 writing-style captions** (formal, sarcastic, humorous_tech, humorous_non_tech) with a text LLM. Output is a `results.json` of `{task_id, captions}`.

The codebase is **functional and coherent at the core**, but it carries **significant documentation drift, dead/legacy code, and a few correctness and security issues** (detailed in §7–§8). The actual execution path differs from what `ARCHITECTURE.md`/`PROJECT_DOCUMENTATION.md` describe (e.g. Hugging Face client is documented as the engine but is no longer wired into the main flow; the PDF generator referenced in the docs does not exist).

---

## 2. Repository Inventory

| Path | Type | Purpose |
|------|------|---------|
| `main.py` | Module | CLI entrypoint + competition batch driver (`competition_main`). |
| `app.py` | Module | Streamlit web UI (281→817 lines, single large file). |
| `config.py` | Module | Central config: detector params, CLIP, API keys, models, styles. |
| `scene_detector.py` | Module | Adaptive scene-cut detection → `Scene` dataclass list. |
| `frame_sampler.py` | Module | Frame-sampling strategies + shared `seek_and_read` helper. |
| `frame_embedder.py` | Module | CLIP model load + batch embedding (BGR→RGB→512-d normalized). |
| `frame_selector.py` | Module | Candidate pooling, novelty scoring, farthest-point keyframe selection. |
| `services/` | Package | AI clients (fireworks, gemini, groq, hf), image_analyzer, scene_aggregator, report_generator, prompt_loader, report_cache. |
| `prompts/*.txt` | Data | 4 style templates (formal, sarcastic, humorous_tech, humorous_non_tech). |
| `run_fireworks_reports.py` / `run_gemini_reports.py` | Scripts | Standalone captioning scripts using hardcoded local frame dirs. |
| `tasks.json`, `input.json`, `12ivdeos.json` | Data | Competition task manifests (video URLs + styles). |
| `results.json` | Data | Example competition output (12 tasks, 4 styles each). |
| `Dockerfile`, `requirements.txt`, `requirements-docker.txt`, `.dockerignore` | Build | Containerized deployment (CPU torch, entrypoint = `competition_main`). |
| `ARCHITECTURE.md`, `PROJECT_DOCUMENTATION.md`, `IMPLEMENTATION.md`, `temparch.md` | Docs | Design documentation (partially stale). |
| `video1/2/3.mp4`, `output/` | Assets | Sample videos + generated artifacts. |

---

## 3. Architecture & Data Flow

```
[video.mp4]
   │  Phase A (local, deterministic)
   ▼
detect_scenes() ───────► list[Scene]                (scenedetect.AdaptiveDetector)
   │
   ▼
select_keyframes() ────► list[SelectedFrame]
   ├─ build global candidate pool (CANDIDATE_FPS sampling + scene boundaries)
   ├─ seek_and_read() each candidate frame (cv2)
   ├─ embed_frames() via CLIP ViT-B/32 (512-d L2-normalized)
   ├─ novelty score = 1 - cos(embed[t], embed[t-1])
   └─ greedy farthest-point selection (cap = MAX_FRAMES=15)
   │
   ▼  (saved: keyframe_*.jpg, keyframes.json, scenes.json)
   │  Phase B (online, API)
   ▼
analyze_keyframe() ───► per-frame JSON             (vision LLM: Fireworks/Gemini)
   │
   ▼
aggregate_by_scene() ─► scene-level JSON           (consensus / union / max-severity)
   │
   ▼
generate_*_reports() ─► 4 style captions           (text LLM: Groq/Gemini/Fireworks)
   │
   ▼
results.json  = [ {task_id, captions:{style: text}} ]
```

**Key abstraction:** `select_keyframes(video_path, scenes)` is the single clean handoff point between Phase A and Phase B (used by `main.py`, `app.py`, and `competition_main`).

---

## 4. Phase A — Keyframe Extraction (Detailed)

### 4.1 Scene Detection (`scene_detector.py`)
- Uses `scenedetect.open_video` + `SceneManager` + `AdaptiveDetector`.
- Parameters from `config.DETECTOR_CONFIG`: `adaptive_threshold=3.0`, `min_scene_len=15`, `window_width=2`, `min_content_val=15.0`.
- Returns `list[Scene]` with `scene_number, start_frame, end_frame, start_time, end_time, duration`.
- Logs each scene; robust and simple.

### 4.2 Candidate Sampling (`frame_selector._candidate_frame_numbers`)
- Sampling rate `CANDIDATE_FPS = 5.0` → `step = round(video_fps / 5)`.
- Scene **start and end frames are force-included** even if the step skips them.
- Candidates are pooled **across all scenes**, then de-duplicated by frame index while preserving temporal order. This is a good design: a single-scene and a 20-scene video both bottom out at the same global `MAX_FRAMES` ceiling.

### 4.3 Embedding (`frame_embedder.py`)
- Loads OpenAI CLIP (`clip-anytorch` backend) once and caches it.
- Auto-detects CUDA, falls back to CPU (`_get_device`).
- BGR→RGB conversion, CLIP preprocessing, batched (`EMBEDDING_BATCH_SIZE=32`), L2-normalized to unit vectors → `(N, 512)`.
- `torch.no_grad()` used (correct for inference).

### 4.4 Selection (`frame_selector._farthest_point_selection`)
- Greedy max–min selection starting from candidate index 0 (deterministic).
- Distance computed in Euclidean space; with unit-norm embeddings this equals cosine distance.
- **Early-stop**: if next max–min distance `< EARLY_STOP_MIN_DIST (0.03)`, stop (avoids near-duplicate keyframes on static scenes).
- Hard cap: `budget = min(MAX_FRAMES, n_candidates)`.

### 4.5 Output
- Selected frames downscaled so longest side ≤ `MAX_SAVE_SIDE=1280` (INTER_AREA) before JPEG write — only the on-disk image is downscaled; selection uses full-res frames.
- `keyframes.json` (frame_index, timestamp_sec, scene_id, novelty_score, image_path) and `scenes.json` (HH:MM:SS.mmm timecodes).

**Assessment:** Phase A is well-structured, deterministic, documented, and correct. The main limitation is CPU-only CLIP inference (slow for long videos), though CUDA auto-detection is wired in.

---

## 5. Phase B — AI Captioning (Detailed)

### 5.1 Vision Analysis (`services/image_analyzer.py`)
- Sends each keyframe (base64) + a fixed JSON schema prompt to a vision model.
- `ANALYSIS_PROMPT` requests 13 fields (scene_type, location, people, objects, vehicles, animals, activities, weather, time_of_day, environment, risk_level, confidence, summary).
- Robust JSON parsing: strips ```code fences, regex-extracts `{...}`, falls back to a structured "unknown" object on failure.
- **Important discrepancy:** `analyze_keyframe` accepts a `hf_client` typed as `FireworksClient`, but `main.py` passes either a Fireworks or Gemini client. The call is `hf_client.analyze_image(...)` — both clients implement that method, so it works duck-typed, but the type hints and import (`from services.fireworks_client import FireworksClient`) are misleading for the Gemini path.

### 5.2 Aggregation (`services/scene_aggregator.py`)
- `aggregate_scene_analyses`: majority vote for categorical fields, list-union (dedup) for objects/vehicles/animals, "longest description" for people/summary, **max-severity** for risk_level, **average** confidence.
- `aggregate_by_scene` groups per-frame analyses by `scene_id`.
- Sound heuristic aggregation; edge cases (empty / single analysis) handled.

### 5.3 Report Generation (`services/report_generator.py`)
- 4 system-prompt personas.
- `generate_report` → `generate_all_reports` (parallel via `ThreadPoolExecutor`, max 4 workers).
- `_clean_report_output` strips chain-of-thought from reasoning models, keeps last 2 sentences, caps 300 chars.
- `_local_fallback_report` provides offline templates if the text API fails (good resilience).
- `generate_video_summary_reports` aggregates **all scenes' activities** into one combined prompt → one 2-line caption per style (this is what `competition_main` uses).

### 5.4 Clients (`services/`)
- `fireworks_client.py`: OpenAI-compatible `OpenAI` client; `analyze_image`, `analyze_image_base64`, `generate_text`, `validate_connection`; singleton `get_fireworks_client`.
- `gemini_client.py`: OpenAI-compatible Gemini endpoint; adds `analyze_images_batch` (multi-image single call). Singleton `get_gemini_client`.
- `groq_client.py`: text-only Groq client. Singleton `get_groq_client`.
- `hf_client.py`: `huggingface_hub.InferenceClient` wrapper. **Implemented but NOT used by the main pipeline** (only referenced in stale docs). Legacy/dead code.
- `prompt_loader.py`: loads `prompts/{style}.txt`, substitutes `{scene_data}`, skips metadata keys.
- `report_cache.py`: file-based cache keyed by `scene_{id:03d}/{style}.md` + `cache.json` metadata; `get_cache_stats`.

### 5.5 Execution Modes
| Mode | Trigger | Path |
|------|---------|------|
| Single video | `python main.py video.mp4 [--reports --style/--all-styles --provider]` | `main()` then optional `generate_reports_pipeline` |
| Batch / competition | `python main.py --input tasks.json` or `/input/tasks.json` present | `competition_main()` |
| Web UI | `streamlit run app.py` | `app.py` |

`competition_main` downloads each `video_url`, runs Phase A+B, maps custom styles → supported styles (`STYLE_MAPPING`), and writes `results.json` in the exact competition schema.

---

## 6. Configuration & Build

- **`config.py`** centralizes everything; `load_dotenv()` loads `.env`.
- **API keys**: `FIREWORKS_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `HF_API_TOKEN`; default models point to Fireworks `minimax-m3` / Groq `gpt-oss-120b` and Gemini `gemini-2.5-flash`.
- **`MAX_FRAMES = 15`** (note: docs say 200; `app.py` slider maxes at 50 — three different "truths", see §7).
- **`AI_PROVIDER`** defaults to `"gemini"` in `config.py`, but **`.env.example` sets `AI_PROVIDER=huggingface`** (a provider that the main code no longer supports).
- **Dockerfile**: `python:3.11-slim`, installs CPU torch from the PyTorch index, then `requirements-docker.txt`; entrypoint runs `competition_main()` with `/input` + `/output` mount points. Uses `opencv-python-headless` (correct for containers).

---

## 7. Inconsistencies & Documentation Drift

1. **Docs vs. reality (providers).** `ARCHITECTURE.md` and `PROJECT_DOCUMENTATION.md` describe the pipeline as driven by **Hugging Face InferenceClient** (`hf_client.py`, Qwen2.5-VL / Qwen3-8B). In actual `main.py`, Phase B uses **Fireworks vision + Groq text** (`generate_reports_pipeline`), or Gemini via `app.py`. `hf_client.py` is orphaned dead code.
2. **PDF generator missing.** `ARCHITECTURE.md` §3.2.7 and the data-flow diagram describe `services/pdf_generator.py` (fpdf2) producing `*.pdf` reports. **No such file exists** (only a stale `.pyc` in `__pycache__`). All PDF output is vaporware in docs.
3. **`local_analyzer.py` ghost.** A compiled `local_analyzer.cpython-312.pyc` exists but there is no `local_analyzer.py` source — a deleted/renamed module leaving behind bytecode.
4. **`MAX_FRAMES` mismatch.** Code = `15`; `PROJECT_DOCUMENTATION.md` = `200`; `app.py` slider = `1..50`. Confusing for users.
5. **`AI_PROVIDER` semantics.** `config.AI_PROVIDER` default is `"gemini"`, but `generate_reports_pipeline` **ignores** the `provider` argument for model selection (always Fireworks vision + Groq text); `provider` only ends up in cache metadata. The `--provider` CLI flag and `analyze_keyframe`'s gemini/fireworks branch are effectively bypassed in `main.py`.
6. **`REPORT_STYLES` mismatch.** `config.py` defines 4 styles (no `jargon`); `ARCHITECTURE.md`/some docs list 5 (incl. `jargon`). The `jargon.txt` prompt file does **not** exist. `run_fireworks_reports.py`/`run_gemini_reports.py` hardcode 4 styles too.
7. **`frame_sampler.py` redundancy.** Three sampler strategies (`middle/first/last`) and `extract_frames` exist but are **not invoked** by the active pipeline (selection uses dense sampling instead). Legacy/utility code.
8. **Hardcoded local paths.** `run_fireworks_reports.py` and `run_gemini_reports.py` point at `C:\Users\a\Downloads\Video-Caption-temp1\...` — developer-machine paths, not portable.

---

## 8. Issues, Risks & Bugs

### Security
- **🔴 Secret committed in `.env.example`:** a real-looking `HF_API_TOKEN` was hardcoded. Even if example-only, committing tokens is a bad practice and a leak risk; should be `HF_API_TOKEN=your_token_here` and the value rotated/revoked.
- `AI_PROVIDER=huggingface` in `.env.example` selects an unsupported path.

### Correctness / Robustness
- **Type hint mismatch**: `analyze_keyframe(hf_client: FireworksClient, ...)` is called with Gemini/Groq clients (works by duck typing, but misleading and fragile to refactor).
- **`competition_main` deletes nothing for failed tasks' temp videos on the exception path** (only success path `os.unlink`s). Minor temp-disk leak under failures.
- **`_video_short_name`** derives output folder from text before first `-`; for `1860079-uhd_...mp4` → `1860079`. Fine, but collisions possible for similarly named inputs.
- **No `fpdf2` / `huggingface_hub` in `requirements.txt`**, yet docs reference them; `requirements-docker.txt` also omits `huggingface_hub` (consistent with HF being unused) but **omits `streamlit` and `torch`** (torch installed separately in Dockerfile; streamlit not needed in container — acceptable, but `requirements.txt` lists `streamlit` and `clip-anytorch` while `requirements-docker.txt` drops `torch` explicitly — verify the CPU torch install step matches).
- **Pexels URLs in `tasks.json`** (`https://www.pexels.com/download/...`) may 403/redirect; download relies on `requests` following redirects — untested robustness.
- Duplicate frame-saving/downscale logic is copy-pasted across `main.py`, `app.py`, and the batch section of `app.py` (3 copies) — maintenance hazard.

### Performance
- CLIP runs on CPU by default; for UHD 4K inputs (e.g. `garden_kitten` is 3840×2160) embedding many candidates is slow. `EMBEDDING_BATCH_SIZE=32` is modest. No multiprocessing for embedding.
- Vision + text calls are parallelized (`ThreadPoolExecutor`, max 15 vision workers) — good.

---

## 9. Strengths

- Clean separation: detection / sampling / embedding / selection are small, single-responsibility modules.
- Deterministic, reproducible keyframe selection (seeded farthest-point).
- Good fallback behavior: structured "unknown" JSON, local report templates, scene-data fallback when vision API fails — the pipeline degrades gracefully instead of crashing.
- Retry-with-backoff for rate limits (`retry_api_call`, `429` handling).
- File-based report cache avoids recomputation.
- Dockerized for headless competition execution with correct headless OpenCV.
- Style-mapping layer lets arbitrary competition styles map onto the 4 supported ones.

---

## 10. Recommendations (Prioritized)

1. **🔴 Rotate & scrub the HF token** in `.env.example`; replace with a placeholder. Add `.env` to `.gitignore` if not already.
2. **Reconcile docs with code**: update `ARCHITECTURE.md`/`PROJECT_DOCUMENTATION.md` to reflect Fireworks+Groq (or Gemini) as the real engine; remove PDF-generator and `jargon` references; mark `hf_client.py` as legacy or delete it.
3. **Unify `MAX_FRAMES`** semantics across `config.py`, docs, and `app.py` slider.
4. **Fix provider routing** in `main.py`: either honor `--provider`/`AI_PROVIDER` for model selection or remove the dead branch and clarify that Phase B = Fireworks vision + Groq text.
5. **Remove or clearly mark dead code**: `frame_sampler.extract_frames`/samplers (unused by active path), `hf_client.py`, `run_*_reports.py` (hardcoded paths), and clean the `local_analyzer` pyc.
6. **Extract a shared `save_keyframe` helper** to remove the 3× duplicated downscale/save logic.
7. **Temp-file cleanup** on `competition_main` exception paths.
8. **Add `huggingface_hub` / `fpdf2` to requirements only if actually used**; otherwise drop from docs.
9. Consider GPU/ONNX CLIP or frame-skip for long UHD videos to improve Phase A throughput.

---

## 11. Conclusion

`video-amd-main` is a **capable, competition-ready video→caption pipeline** with a solid, deterministic keyframe-extraction core (scene detection + CLIP + farthest-point selection) and a resilient multi-provider AI captioning layer producing 4 stylistic outputs. The engineering is pragmatic and fault-tolerant.

Its primary weaknesses are **not in runtime correctness but in project hygiene**: documentation that describes an older Hugging Face–centric design, missing referenced modules (PDF generator), committed secrets, and scattered dead/legacy code. Addressing the items in §10—especially the secret leak and the doc/code reconciliation—would bring the repository to a clean, production-grade state.
