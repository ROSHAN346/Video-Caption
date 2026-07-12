# Project Report — video-amd (AI Video Captioning Pipeline)

Date: 2026-07-12

## 1. Overview

video-amd is an AI video captioning pipeline built for an AMD hackathon. Given a
JSON list of video URLs, it downloads each video, detects scenes, selects a small
set of visually diverse keyframes using CLIP embeddings, analyzes them with a
Fireworks vision model, aggregates the results per scene, and generates short
captions in multiple writing styles (formal, sarcastic, humorous_tech,
humorous_non_tech) via a Groq text model. Results are written to `results.json`.

Run modes:

| Mode | Command |
|---|---|
| Batch / competition | `python main.py --input input/tasks.json --output out/results.json` |
| Single video | `python main.py video.mp4 --reports [--style formal \| --all-styles]` |
| Docker | auto-runs batch mode against `/input/*.json` → `/output/results.json` |
| Web UI | `streamlit run app.py` |

## 2. Architecture

```
tasks.json → download_video → detect_scenes ─┬→ select_keyframes (CLIP + farthest-point)
                                             └→ captured frames (single decode pass)
           → analyze_keyframe_array (Fireworks vision, threaded, retry/backoff)
           → aggregate_by_scene (consensus voting)
           → generate_video_summary_reports (Groq text, per-style, parallel)
           → results.json
```

Key modules:

- `scene_detector.py` — PySceneDetect AdaptiveDetector driven in a single decode
  pass; candidate frames captured in memory (uniform grid at `CANDIDATE_FPS` +
  scene boundaries), downscaled to `MAX_CAPTURE_SIDE` to bound RAM.
- `frame_selector.py` — greedy farthest-point selection over CLIP embeddings,
  capped at `MAX_FRAMES=5`, with early-stop on near-duplicate candidates.
- `frame_embedder.py` — CLIP ViT-B/32; auto device select (CUDA / DirectML / CPU).
- `services/` — Fireworks vision client, Groq text client, structured frame
  analysis, per-scene aggregation, styled report generation, prompt loading,
  file-based report cache.

Design strengths:

- Video decoded exactly once; keyframes taken from memory.
- Global keyframe budget keeps vision API cost fixed per video (max 5 calls).
- Graceful degradation: placeholder analyses/captions if APIs fail, so batch
  output is always complete.
- Style mapping tolerates unknown style names in input tasks.

## 3. Issues found and fixed

### Security
| Issue | Fix |
|---|---|
| Real Fireworks + Groq API keys shipped in `.env` | Deleted; only `.env.example` remains. **Keys must be rotated.** |
| `ANTHROPIC_API_KEY` baked into the published Docker image ENV (`silver-octo-guacamole:latest`) | Documented; keys now passed at runtime only (`-e` / `--env-file`). **Key must be revoked.** |
| `app.py` executed a user-supplied "Python interpreter" string (arbitrary command execution) | Field removed; uses `sys.executable`. |
| `download_video()` fetched arbitrary URLs with no limits | Added 1 GB cap (`MAX_DOWNLOAD_MB`), HTML content-type rejection, connect+read timeouts, partial-file cleanup. |

### Bugs
| Issue | Fix |
|---|---|
| `EARLY_STOP_MIN_DIST` defined but never passed → early-stop dead | Now passed to `_farthest_point_selection` (frame_selector.py). |
| Full-resolution frames held in memory for whole video (GBs on UHD) | Captured frames downscaled to `MAX_CAPTURE_SIDE=1280` (~4–10x RAM reduction, no quality loss downstream). |
| `parse_args()` called twice | `main(args)` receives parsed args. |
| Task IDs with trailing whitespace propagated into output | `task_id` stripped. |
| `scenedetect` pin inconsistency (`>=0.5.0` vs `0.7.*`) while using 0.7 private APIs | Pinned `==0.7.*` everywhere. |

### Dead code removed
- `frame_sampler.py` (superseded by embedding-based selection)
- Path-based `analyze_keyframe` / `analyze_keyframes_batch`
- Unused `FireworksClient.analyze_image` / `.generate_text` / `.validate_connection`
- `AI_PROVIDER`, `--provider`, `FIREWORKS_TEXT_MODEL`, `FRAME_STRATEGY`

### Repository hygiene
- Removed: `__pycache__/` (incl. stale compiled modules), `video1-3.mp4`,
  `12ivdeos.json`, duplicate root `results.json`, scratch files
  (`run_fireworks_reports.py`, `temparch.md`, `final.md`)
- 4 overlapping/stale docs consolidated into one accurate `README.md`
- `.gitignore` fixed (previously ignored all `.md`/`.json` and `.env.example`)
- `.dockerignore` tightened (excludes docs, UI, secrets, caches)

### Docker
| Issue | Fix |
|---|---|
| Build-time `/dev/kfd` GPU check (never true during `docker build`) | Explicit `--build-arg TORCH_FLAVOR=cpu\|rocm`. |
| CLIP weights deleted after build → ~350 MB re-download every cold start, breaks offline | Weights kept in image. |
| `requirements-docker.txt` copied but never installed (duplicated inline lists) | Installs from `requirements-docker.txt` + `requirements-docker-nodeps.txt`. |
| Entrypoint hardcoded `/input/tasks.json`, bypassing fallback resolution | Passes `None`; `/input/tasks.json`, `/input/input.json`, or any `/input/*.json` accepted. |

## 4. Final structure

```
video-amd/
├── main.py, app.py, config.py
├── scene_detector.py, frame_selector.py, frame_embedder.py
├── services/            (7 modules)
├── prompts/             (4 style templates)
├── input/tasks.json     (sample)
├── Dockerfile, docker-entrypoint.sh
├── requirements.txt, requirements-docker.txt, requirements-docker-nodeps.txt
├── .env.example, .gitignore, .dockerignore
└── README.md, PROJECT_REPORT.md
```

## 5. Verification

- `python -m py_compile` passes for all 14 modules
- All modules import cleanly (`IMPORTS OK`)
- CLI arg-validation path behaves correctly
- No references to removed symbols remain (grep-verified)

## 6. Outstanding recommendations

1. **Rotate/revoke all three exposed API keys** (Fireworks, Groq, Anthropic) — cannot be done from code.
2. Rebuild and republish the Docker image without keys in ENV.
3. Consider replacing PySceneDetect private-API usage (`_process_frame`, `_cutting_list`) with the public API when upgrading beyond 0.7.x.
4. Optional: add unit tests (scene detection on synthetic clips, style mapping, JSON parsing fallbacks) and CI.
