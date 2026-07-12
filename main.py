import sys
import json
import os
import time
import re
import shutil
import logging
import argparse
import urllib.request
import urllib.error
import cv2
from pathlib import Path

from scene_detector import detect_scenes
from frame_selector import select_keyframes
from config import (
    FRAME_STRATEGY, MAX_FRAMES, REPORT_STYLES,
    DEFAULT_REPORT_STYLE, HF_API_TOKEN, HF_VISION_MODEL, HF_TEXT_MODEL,
    FIREWORKS_API_KEY, FIREWORKS_VISION_MODEL, FIREWORKS_TEXT_MODEL, AI_PROVIDER
)

# Saved keyframes are downscaled to cap artifact size (UHD sources otherwise
# produce huge JPEGs). Selection/embeddings use the full-quality frame, so this
# only affects the on-disk JPEG, not keyframe quality.
MAX_SAVE_SIDE = 1280


def _downscale_to_max_side(frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= MAX_SAVE_SIDE:
        return frame
    scale = MAX_SAVE_SIDE / float(long)
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Keyframe Extraction with AI Report Generation"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to a single video file (mutually exclusive with --input)",
    )
    parser.add_argument(
        "--input",
        help="Batch JSON task list (each entry: task_id, video_url, styles)",
    )
    parser.add_argument(
        "--output",
        help="Path to write results.json (used with --input)",
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate AI analysis reports",
    )
    parser.add_argument(
        "--style",
        choices=REPORT_STYLES,
        default=DEFAULT_REPORT_STYLE,
        help=f"Report style (default: {DEFAULT_REPORT_STYLE})",
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Generate reports in all styles (single-video mode only)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation",
    )
    parser.add_argument(
        "--provider",
        choices=["huggingface", "fireworks"],
        default=AI_PROVIDER,
        help=f"AI provider to use (default: {AI_PROVIDER})",
    )

    args = parser.parse_args()

    if args.input and args.video_path:
        parser.error("Cannot specify both a positional video_path and --input")
    if args.input and not args.output:
        parser.error("--output is required when --input is used")
    if args.output and not args.input:
        parser.error("--output requires --input")

    return args


def generate_reports_pipeline(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str],
    generate_pdf: bool = True,
    provider: str = AI_PROVIDER,
) -> dict:
    """Run the per-scene report generation pipeline.

    Writes per-scene .md (and optionally .pdf) reports under
    ``output_dir/reports/scene_XXX/<style>.{md,pdf}``.

    Returns:
        ``{"analyses": [...], "scene_analyses": {...}}`` so the caller can
        synthesize a video-level summary without re-running analysis.
    """
    from services.hf_client import get_hf_client
    from services.fireworks_client import get_fireworks_client
    from services.image_analyzer import analyze_keyframe, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_all_reports
    from services.report_cache import ReportCache
    from services.pdf_generator import generate_report_pdf

    empty = {"analyses": [], "scene_analyses": {}}

    logger.info(f"=== START Report Generation (provider: {provider}) ===")

    try:
        if provider == "fireworks":
            if not FIREWORKS_API_KEY:
                logger.error(
                    "FIREWORKS_API_KEY not set. Please create a .env file with your key.\n"
                    "Get yours at: https://fireworks.ai/account/api-keys"
                )
                return empty
            client = get_fireworks_client()
            vision_model = FIREWORKS_VISION_MODEL
            text_model = FIREWORKS_TEXT_MODEL
            logger.info("Using Fireworks AI provider")
        else:
            if not HF_API_TOKEN:
                logger.error(
                    "HF_API_TOKEN not set. Please create a .env file with your token.\n"
                    "Get yours at: https://huggingface.co/settings/tokens"
                )
                return empty
            client = get_hf_client()
            vision_model = HF_VISION_MODEL
            text_model = HF_TEXT_MODEL
            logger.info("Using Hugging Face provider")
    except Exception as e:
        logger.error(f"Failed to initialize {provider} client: {e}")
        return empty

    cache = ReportCache(str(output_dir / "reports"))

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing {len(selected)} keyframes with vision model...")
    analyses = []

    for i, sf in enumerate(selected):
        keyframe_path = str(output_dir / f"keyframe_{i:03d}.jpg")

        if not Path(keyframe_path).exists():
            logger.warning(f"Keyframe not found: {keyframe_path}")
            continue

        try:
            analysis = analyze_keyframe(
                image_path=keyframe_path,
                hf_client=client,
                model=vision_model,
                scene_id=sf.scene_id,
                frame_index=i,
            )
            if analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
                logger.warning(f"Frame {i} analysis failed, skipping")
                continue
            analyses.append(analysis)

            analysis_path = analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
            save_analysis(analysis, str(analysis_path))

            logger.info(f"Analyzed frame {i}: {analysis.get('summary', 'N/A')[:50]}...")
        except Exception as e:
            logger.warning(f"Frame {i} failed: {e}, skipping")
            continue

    if not analyses:
        logger.warning("No keyframes analyzed via vision model. Using scene detection data as fallback.")
        for scene in scenes:
            fallback = {
                "scene_id": scene.scene_number,
                "frame_index": 0,
                "scene_type": "video content",
                "location": "detected scene",
                "people": "not analyzed",
                "objects": ["video frames"],
                "vehicles": [],
                "animals": [],
                "activities": f"scene spanning {scene.duration:.1f} seconds",
                "weather": "unknown",
                "time_of_day": "unknown",
                "environment": "unknown",
                "risk_level": "low",
                "confidence": 0.5,
                "summary": f"Scene {scene.scene_number}: {scene.duration:.1f}s video segment (frames {scene.start_frame}-{scene.end_frame}) with {len(selected)} keyframes selected for analysis.",
            }
            analyses.append(fallback)
        logger.info(f"Created fallback analysis for {len(scenes)} scenes")

    logger.info("Aggregating analyses by scene...")
    scene_analyses = aggregate_by_scene(analyses, scenes)

    logger.info(f"Generating {len(styles)} report styles...")
    for scene_id, scene_data in scene_analyses.items():
        logger.info(f"Processing scene {scene_id}...")

        reports = generate_all_reports(scene_data, client, styles, text_model)

        for style, report in reports.items():
            cache.cache_report(scene_id, style, report, {
                "vision_model": vision_model,
                "text_model": text_model,
                "provider": provider,
            })

            md_path = cache.get_cache_path(scene_id, style)
            logger.info(f"Saved {style} report: {md_path}")

            if generate_pdf:
                keyframe_path = str(output_dir / "keyframe_000.jpg")
                pdf_path = cache.get_pdf_path(scene_id, style)
                try:
                    generate_report_pdf(
                        scene_id=scene_id,
                        style=style,
                        report=report,
                        keyframe_path=keyframe_path,
                        scene_data=scene_data,
                        output_path=str(pdf_path),
                    )
                    logger.info(f"Generated PDF: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error generating PDF for {style}: {e}")

    stats = cache.get_cache_stats()
    logger.info("=== Report Generation Complete ===")
    logger.info(f"Scenes processed: {stats['total_scenes']}")
    logger.info(f"Reports generated: {stats['total_reports']}")
    logger.info(f"Output directory: {output_dir / 'reports'}")

    return {"analyses": analyses, "scene_analyses": scene_analyses}


def _generate_video_summary(
    scene_analyses: dict,
    styles: list[str],
    provider: str,
) -> dict[str, str]:
    """Synthesize one video-level summary text per requested style.

    Uses ``generate_video_summary_reports`` which collapses
    ``aggregate_activities`` over all scenes into a single prompt per style.
    Falls back to a stub if there are no scene analyses.
    """
    if not scene_analyses:
        return {style: "Report unavailable (no scene analyses)" for style in styles}

    from services.hf_client import get_hf_client
    from services.fireworks_client import get_fireworks_client
    from services.report_generator import generate_video_summary_reports

    try:
        client = (
            get_fireworks_client() if provider == "fireworks" else get_hf_client()
        )
        text_model = (
            FIREWORKS_TEXT_MODEL if provider == "fireworks" else HF_TEXT_MODEL
        )
    except Exception as e:
        logger.error(f"Failed to build summary client: {e}")
        return {style: f"Report unavailable (client init failed: {e})" for style in styles}

    try:
        return generate_video_summary_reports(
            scene_analyses=scene_analyses,
            hf_client=client,
            styles=styles,
            model=text_model,
        )
    except Exception as e:
        logger.error(f"Video-level summary failed: {e}")
        return {style: f"Report unavailable ({e})" for style in styles}


def format_timecode(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _video_short_name(video_path: str) -> str:
    """Derive a short, filesystem-safe output folder name from the video file.

    Uses the text before the first '-' in the filename stem (falls back to the
    full stem when there is no '-'), so '1860079-uhd_...mp4' -> '1860079' and
    'traffic.mp4' -> 'traffic'.
    """
    stem = Path(video_path).stem
    prefix = stem.split("-", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", prefix).strip("._-")
    return safe or stem


def _safe_task_id(task_id: str) -> str:
    """Sanitize a task_id for filesystem use (e.g. dict key, output folder)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", task_id).strip("._-")
    return safe or "task"


def _infer_ext_from_url(url: str) -> str:
    """Best-effort extension from a URL (ignores query string)."""
    path = url.split("?", 1)[0].lower()
    for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        if path.endswith(ext):
            return ext
    return ".mp4"


def _download_video(url: str, dest: Path) -> None:
    """Stream-download a video URL to ``dest``. Skips if already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Cache hit ({dest.stat().st_size} bytes): {dest}")
        return

    logger.info(f"Downloading: {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e

    logger.info(f"Downloaded {dest.stat().st_size} bytes -> {dest}")


def _ensure_video(task_id: str, url: str) -> Path:
    """Return a local path for the task video, downloading if not cached."""
    safe = _safe_task_id(task_id)
    ext = _infer_ext_from_url(url)
    dest = Path("input") / "videos" / f"{safe}{ext}"
    _download_video(url, dest)
    return dest


def _load_tasks(path: str) -> list[dict]:
    """Load the batch task JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _build_ai_client(provider: str):
    """Construct the configured AI client (HF or Fireworks)."""
    if provider == "fireworks":
        if not FIREWORKS_API_KEY:
            raise RuntimeError("FIREWORKS_API_KEY not set in .env")
        from services.fireworks_client import get_fireworks_client
        return get_fireworks_client()
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in .env")
    from services.hf_client import get_hf_client
    return get_hf_client()


def _run_pipeline(
    video_path: str,
    output_dir: Path,
    ai_client,
    styles: list[str],
    provider: str,
    generate_pdf: bool,
) -> dict:
    """Run the full per-video pipeline and return an in-memory result dict.

    Side effects:
        - Writes JPEGs, ``keyframes.json``, ``scenes.json`` under ``output_dir``.
        - When ``styles`` is non-empty, writes per-scene reports and PDFs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    logger.info(f"=== START pipeline for: {video_path} ===")

    scenes = detect_scenes(video_path)
    logger.info(f"Number of scenes found: {len(scenes)}")
    logger.info(f"Output folder: {output_dir.resolve()}")

    selected = select_keyframes(video_path, scenes)
    logger.info(f"Selected keyframes: {len(selected)} (MAX_FRAMES budget enforced globally)")

    keyframes_data = []
    save_start = time.time()
    for i, sf in enumerate(selected):
        filename = f"keyframe_{i:03d}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), _downscale_to_max_side(sf.image))
        keyframes_data.append({
            "frame_index": sf.frame_index,
            "timestamp_sec": round(sf.timestamp_sec, 3),
            "scene_id": sf.scene_id,
            "novelty_score": round(sf.novelty_score, 4),
            "image_path": filename,
        })
        logger.info(
            f"Saved: {filename} | scene={sf.scene_id} "
            f"frame={sf.frame_index} t={sf.timestamp_sec:.2f}s "
            f"novelty={sf.novelty_score:.3f}"
        )
    logger.info(f"Wrote {len(keyframes_data)} JPEGs in {time.time() - save_start:.2f}s")

    keyframes_json_path = output_dir / "keyframes.json"
    with open(keyframes_json_path, "w") as f:
        json.dump(keyframes_data, f, indent=2)
    logger.info(f"Saved: keyframes.json ({len(keyframes_data)} frames)")

    scenes_data = []
    for scene in scenes:
        scenes_data.append({
            "scene_number": scene.scene_number,
            "start_time": format_timecode(scene.start_time),
            "end_time": format_timecode(scene.end_time),
            "duration": scene.duration,
        })
    scenes_json_path = output_dir / "scenes.json"
    with open(scenes_json_path, "w") as f:
        json.dump(scenes_data, f, indent=2)
    logger.info(f"Saved: scenes.json ({len(scenes_data)} scenes)")

    duration_sec = round(sum(s.duration for s in scenes), 3)
    total = time.time() - run_start
    logger.info(
        f"Pipeline Phase complete: scenes={len(scenes)} | keyframes={len(selected)} "
        f"(<=MAX_FRAMES={MAX_FRAMES}) | total runtime={total:.2f}s"
    )

    result = {
        "scenes": scenes_data,
        "keyframes": keyframes_data,
        "duration_sec": duration_sec,
        "analyses": [],
        "scene_analyses": {},
        "reports_by_style": {},
    }

    if styles and ai_client is not None:
        report_start = time.time()
        report_bundle = generate_reports_pipeline(
            output_dir=output_dir,
            selected=selected,
            scenes=scenes,
            styles=styles,
            generate_pdf=generate_pdf,
            provider=provider,
        )
        result["analyses"] = report_bundle["analyses"]
        result["scene_analyses"] = report_bundle["scene_analyses"]
        result["reports_by_style"] = _generate_video_summary(
            scene_analyses=report_bundle["scene_analyses"],
            styles=styles,
            provider=provider,
        )
        report_total = time.time() - report_start
        logger.info(f"Report generation completed in {report_total:.2f}s")

    return result


def _process_task(task: dict, ai_client, no_pdf: bool) -> dict:
    """Process one batch task and return its result record (or error record)."""
    raw_task_id = str(task.get("task_id", "")).strip() or "task"
    safe_id = _safe_task_id(raw_task_id)
    video_url = task.get("video_url", "")
    styles = task.get("styles") or REPORT_STYLES

    record: dict = {
        "task_id": raw_task_id,
        "video_url": video_url,
        "video_path": None,
        "duration_sec": None,
        "scenes": [],
        "keyframes": [],
        "reports_by_style": {},
        "error": None,
    }

    try:
        video_path = _ensure_video(raw_task_id, video_url)
        record["video_path"] = str(video_path)

        output_dir = Path("output") / safe_id
        result = _run_pipeline(
            video_path=str(video_path),
            output_dir=output_dir,
            ai_client=ai_client,
            styles=styles,
            provider=AI_PROVIDER,
            generate_pdf=not no_pdf,
        )
        record["duration_sec"] = result["duration_sec"]
        record["scenes"] = result["scenes"]
        record["keyframes"] = result["keyframes"]
        record["reports_by_style"] = result["reports_by_style"]
    except Exception as e:
        logger.exception(f"Task {raw_task_id!r} failed: {e}")
        record["error"] = f"{type(e).__name__}: {e}"

    return record


def _run_batch(args) -> None:
    """Batch entrypoint: iterate tasks, run pipeline, write results.json."""
    tasks = _load_tasks(args.input)
    logger.info(f"Loaded {len(tasks)} task(s) from {args.input}")

    ai_client = None
    try:
        ai_client = _build_ai_client(args.provider)
    except Exception as e:
        logger.error(f"AI client init failed; report sections will be empty: {e}")

    results: dict[str, dict] = {}
    for i, task in enumerate(tasks, start=1):
        raw_task_id = str(task.get("task_id", "")).strip() or f"task_{i}"
        safe_id = _safe_task_id(raw_task_id)
        logger.info(f"=== [{i}/{len(tasks)}] task_id={raw_task_id!r} ===")
        record = _process_task(task, ai_client, args.no_pdf)
        results[safe_id] = record

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote results ({len(results)} task(s)): {out_path.resolve()}")


def _run_single(args) -> None:
    """Single-video entrypoint: preserves original CLI behavior."""
    if not args.video_path:
        print("Error: video_path is required (or pass --input for batch mode)", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.video_path):
        print(f"Error: Video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    if args.all_styles:
        styles = REPORT_STYLES
    else:
        styles = [args.style]

    ai_client = None
    if args.reports:
        try:
            ai_client = _build_ai_client(args.provider)
        except Exception as e:
            logger.error(f"AI client init failed; reports will be skipped: {e}")

    output_dir = Path("output") / _video_short_name(args.video_path)
    _run_pipeline(
        video_path=args.video_path,
        output_dir=output_dir,
        ai_client=ai_client,
        styles=styles if args.reports else [],
        provider=args.provider,
        generate_pdf=not args.no_pdf,
    )
    logger.info(f"Output: {output_dir.resolve()}")


def main():
    args = parse_args()
    if args.input:
        _run_batch(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
