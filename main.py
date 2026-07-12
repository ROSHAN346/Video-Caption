import sys
import json
import os
import time
import re
import logging
import argparse
import tempfile
import requests
import cv2
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,
)

from scene_detector import detect_scenes
from frame_selector import select_keyframes
from config import (
    MAX_FRAMES, REPORT_STYLES,
    DEFAULT_REPORT_STYLE, FIREWORKS_API_KEY, FIREWORKS_VISION_MODEL,
    FIREWORKS_TEXT_API_KEY, FIREWORKS_TEXT_MODEL,
    MAX_DOWNLOAD_MB, MAX_TASK_WORKERS, DEADLINE_SECONDS,
)
from services.fireworks_client import get_fireworks_client, get_fireworks_text_client

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
    parser.add_argument("video_path", nargs="?", help="Path to video file")
    parser.add_argument(
        "--input",
        dest="input_json",
        help="Path to input tasks.json for competition/batch mode"
    )
    parser.add_argument(
        "--output",
        dest="output_json",
        default="results.json",
        help="Path to output results.json (default: results.json)"
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate AI analysis reports"
    )
    parser.add_argument(
        "--style",
        choices=REPORT_STYLES,
        default=DEFAULT_REPORT_STYLE,
        help=f"Report style (default: {DEFAULT_REPORT_STYLE})"
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Generate reports in all styles"
    )
    return parser.parse_args()


def generate_reports_pipeline(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str],
):
    """Run the report generation pipeline: Fireworks vision + Fireworks text."""
    from services.image_analyzer import analyze_keyframe_array, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_all_reports
    from services.report_cache import ReportCache

    logger.info("=== START Report Generation (Fireworks vision + Fireworks text) ===")

    # Initialize vision client (Fireworks key #1) + text client (Fireworks key #2)
    try:
        if not FIREWORKS_API_KEY:
            logger.error("FIREWORKS_API_KEY not set in .env")
            return
        if not FIREWORKS_TEXT_API_KEY:
            logger.error("FIREWORKS_TEXT_API_KEY not set in .env")
            return
        vision_client = get_fireworks_client()
        text_client = get_fireworks_text_client()
        vision_model = FIREWORKS_VISION_MODEL
        text_model = FIREWORKS_TEXT_MODEL
        logger.info(f"Vision: Fireworks ({vision_model}) | Text: Fireworks ({text_model})")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return

    cache = ReportCache(str(output_dir / "reports"))

    # Create analysis directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Analyze keyframes in parallel
    logger.info(f"Analyzing {len(selected)} keyframes with vision model in parallel...")
    analyses = []

    def _analyze_keyframe_pipeline(i, sf):
        try:
            analysis = analyze_keyframe_array(
                image=sf.image,
                hf_client=vision_client,
                model=vision_model,
                scene_id=sf.scene_id,
                frame_index=i
            )
            if analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
                logger.warning(f"Frame {i} analysis failed, skipping")
                return None
            analysis_path = analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
            save_analysis(analysis, str(analysis_path))
            logger.info(f"Analyzed frame {i}: {analysis.get('summary', 'N/A')[:50]}...")
            return analysis
        except Exception as e:
            logger.warning(f"Frame {i} failed: {e}, skipping")
            return None

    with ThreadPoolExecutor(max_workers=min(len(selected), 15)) as executor:
        futures = {executor.submit(_analyze_keyframe_pipeline, i, sf): i for i, sf in enumerate(selected)}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                analyses.append(result)

    logger.info(f"Analyzed {len(analyses)}/{len(selected)} keyframes successfully")

    if not analyses:
        logger.warning("No keyframes analyzed via vision model. Using scene detection data as fallback.")
        # Create fallback analysis from scene detection data
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
                "summary": f"Scene {scene.scene_number}: {scene.duration:.1f}s video segment (frames {scene.start_frame}-{scene.end_frame}) with {len(selected)} keyframes selected for analysis."
            }
            analyses.append(fallback)
        logger.info(f"Created fallback analysis for {len(scenes)} scenes")

    # Aggregate by scene
    logger.info("Aggregating analyses by scene...")
    scene_analyses = aggregate_by_scene(analyses, scenes)

    # Generate reports
    logger.info(f"Generating {len(styles)} report styles...")
    all_captions = {}
    for scene_id, scene_data in scene_analyses.items():
        logger.info(f"Processing scene {scene_id}...")

        reports = generate_all_reports(scene_data, text_client, styles, text_model)

        for style, report in reports.items():
            # Cache report
            cache.cache_report(scene_id, style, report, {
                "vision_model": vision_model,
                "text_model": text_model,
                "provider": "fireworks"
            })

            # Save markdown report
            md_path = cache.get_cache_path(scene_id, style)
            logger.info(f"Saved {style} report: {md_path}")

            # Collect captions
            all_captions[style] = report

    # Print summary
    stats = cache.get_cache_stats()
    logger.info("=== Report Generation Complete ===")
    logger.info(f"Scenes processed: {stats['total_scenes']}")
    logger.info(f"Reports generated: {stats['total_reports']}")
    logger.info(f"Output directory: {output_dir / 'reports'}")

    return all_captions


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


def download_video(url: str, timeout: int = 120) -> str:
    """
    Download a video from a URL to a temporary file.

    Enforces a hard size cap (MAX_DOWNLOAD_MB) and rejects obviously
    non-video responses (HTML error pages, etc.) to protect disk and
    avoid feeding garbage into the decoder.

    Args:
        url: Video URL
        timeout: Download timeout in seconds (connect + read)

    Returns:
        Path to downloaded video file
    """
    max_bytes = MAX_DOWNLOAD_MB * 1024 * 1024
    logger.info(f"Downloading video from: {url}")
    response = requests.get(url, timeout=(10, timeout), stream=True)
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type.startswith("text/html"):
        response.close()
        raise ValueError(f"URL returned HTML, not a video (Content-Type: {content_type}): {url}")

    declared = response.headers.get("Content-Length")
    if declared and int(declared) > max_bytes:
        response.close()
        raise ValueError(
            f"Video too large: {int(declared) / 1024 / 1024:.0f} MB > {MAX_DOWNLOAD_MB} MB cap: {url}"
        )

    # Determine file extension from URL
    suffix = ".mp4"
    url_path = url.split("?")[0]
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        if url_path.lower().endswith(ext):
            suffix = ext
            break

    written = 0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            for chunk in response.iter_content(chunk_size=64 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"Download exceeded {MAX_DOWNLOAD_MB} MB cap, aborting: {url}"
                    )
                tmp_file.write(chunk)
    except Exception:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    finally:
        response.close()

    logger.info(f"Downloaded video to: {tmp_path} ({written / 1024 / 1024:.1f} MB)")
    return tmp_path


def retry_api_call(func, max_retries: int = 3, base_delay: float = 5.0):
    """
    Retry an API call with exponential backoff.

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)

    Returns:
        Result of func() or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                logger.error(f"API call failed: {e}")
                return None
    logger.error(f"API call failed after {max_retries} retries")
    return None


def generate_captions_for_video(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str]
) -> dict[str, str]:
    """
    Generate video-level captions for all requested styles.
    Uses Fireworks vision (key #1) + Fireworks text (key #2).
    """
    from services.image_analyzer import analyze_keyframe_array, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_video_summary_reports

    if not FIREWORKS_API_KEY:
        logger.error("FIREWORKS_API_KEY not set in .env")
        return {}
    if not FIREWORKS_TEXT_API_KEY:
        logger.error("FIREWORKS_TEXT_API_KEY not set in .env")
        return {}

    vision_client = get_fireworks_client()
    text_client = get_fireworks_text_client()
    vision_model = FIREWORKS_VISION_MODEL
    text_model = FIREWORKS_TEXT_MODEL

    # Analyze keyframes in parallel
    logger.info(f"Analyzing {len(selected)} keyframes in parallel...")
    analyses = []
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _analyze_one(i, sf):
        try:
            analysis = retry_api_call(
                lambda img=sf.image, s=sf, idx=i: analyze_keyframe_array(
                    image=img,
                    hf_client=vision_client,
                    model=vision_model,
                    scene_id=s.scene_id,
                    frame_index=idx
                )
            )
            if analysis and analysis.get("summary") not in ["Analysis unavailable", "Analysis failed to parse", ""]:
                save_analysis(analysis, str(analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"))
                logger.info(f"Analyzed frame {i}: {analysis.get('summary', 'N/A')[:50]}...")
                return analysis
        except Exception as e:
            logger.warning(f"Frame {i} failed: {e}")
        return None

    with ThreadPoolExecutor(max_workers=min(len(selected), 15)) as executor:
        futures = {executor.submit(_analyze_one, i, sf): i for i, sf in enumerate(selected)}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                analyses.append(result)

    logger.info(f"Analyzed {len(analyses)}/{len(selected)} keyframes successfully")

    # Fallback: create placeholder analyses from scene data if no API analyses succeeded
    if not analyses:
        logger.warning("No API analyses succeeded. Using scene data as fallback.")
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
                "summary": f"Scene {scene.scene_number}: {scene.duration:.1f}s video segment with {len(selected)} keyframes."
            }
            analyses.append(fallback)

    # Aggregate by scene
    scene_analyses = aggregate_by_scene(analyses, scenes)

    # Generate video-level captions
    logger.info(f"Generating {len(styles)} caption styles...")
    captions = retry_api_call(
        lambda: generate_video_summary_reports(scene_analyses, text_client, styles, text_model)
    )

    # Guarantee all styles have captions
    if not captions:
        captions = {}
    for style in styles:
        if style not in captions or not captions[style]:
            captions[style] = f"Video content analyzed: {', '.join(s.get('activities', 'unknown') for s in scene_analyses.values())}"

    return captions


def _resolve_input_path(input_path=None) -> Path:
    """Locate the competition tasks file inside the container.

    Resolution order:
      1. Explicit path (must exist, else hard error).
      2. /input/tasks.json   (convention used by the Dockerfile).
      3. /input/input.json   (common alternative filename).
      4. First *.json found in /input (mount-the-whole-folder case).

    This makes the container tolerant of how the host directory is mounted,
    e.g. `-v ./input:/input` or `-v ./input.json:/input/input.json`.
    """
    if input_path:
        p = Path(input_path)
        if p.exists():
            return p
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    candidates = [Path("/input/tasks.json"), Path("/input/input.json")]
    for c in candidates:
        if c.exists():
            logger.info(f"Using input file: {c}")
            return c

    input_dir = Path("/input")
    if input_dir.is_dir():
        jsons = sorted(input_dir.glob("*.json"))
        if jsons:
            logger.info(f"Using input file: {jsons[0]}")
            return jsons[0]

    logger.error(
        "Input file not found. Expected one of: /input/tasks.json, "
        "/input/input.json, or any *.json inside /input. "
        "Mount your tasks file/folder to /input (e.g. "
        "-v ./input:/input or -v ./input.json:/input/tasks.json)."
    )
    sys.exit(1)


def competition_main(input_path=None, output_path=None):
    """
    Competition entrypoint: read tasks.json, process each task,
    write results.json.
    """
    input_path = _resolve_input_path(input_path)
    output_path = Path(output_path or "/output/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        tasks = json.load(f)

    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")

    # Map custom styles to supported styles
    STYLE_MAPPING = {
        "casual": "formal",
        "motivational": "sarcastic",
        "storytelling": "humorous_tech",
        "professional": "formal",
        "technical": "humorous_tech",
        "educational": "formal",
        "funny": "humorous_non_tech",
        "friendly": "formal",
        "dramatic": "sarcastic",
        "cinematic": "humorous_tech",
        "minimalist": "formal",
        "witty": "sarcastic",
        "analytical": "formal",
        "playful": "humorous_non_tech",
        "informative": "formal",
    }

    def map_styles(styles: list[str]) -> list[str]:
        """Map custom styles to supported styles, removing duplicates."""
        mapped = []
        seen = set()
        for style in styles:
            mapped_style = STYLE_MAPPING.get(style, style)
            if mapped_style not in seen:
                mapped.append(mapped_style)
                seen.add(mapped_style)
        return mapped

    total_start = time.time()
    deadline = total_start + DEADLINE_SECONDS

    # Pre-fill every task with placeholder captions so the output is ALWAYS
    # complete, even if the deadline fires while tasks are still running.
    task_order = []
    results_by_id = {}
    task_styles = {}
    for task in tasks:
        task_id = str(task["task_id"]).strip()
        styles = map_styles(task.get("styles", REPORT_STYLES))
        task_order.append(task_id)
        task_styles[task_id] = styles
        results_by_id[task_id] = {
            style: "Processing did not complete for this video clip."
            for style in styles
        }

    def _process_task(task) -> tuple[str, dict]:
        """Full pipeline for one task. Returns (task_id, captions)."""
        task_id = str(task["task_id"]).strip()
        video_url = task["video_url"]
        styles = task_styles[task_id]
        logger.info(f"=== Processing task {task_id} === styles={styles}")
        task_start = time.time()
        video_path = None
        try:
            video_path = download_video(video_url)

            scenes, captured = detect_scenes(video_path)
            logger.info(f"Task {task_id}: Found {len(scenes)} scenes")

            selected = select_keyframes(video_path, scenes, captured)
            del captured  # free frame memory before the API stage
            logger.info(f"Task {task_id}: Selected {len(selected)} keyframes")

            output_dir = Path(tempfile.mkdtemp(prefix=f"task_{task_id}_"))
            (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
            for i, sf in enumerate(selected):
                cv2.imwrite(str(output_dir / f"keyframe_{i:03d}.jpg"),
                            _downscale_to_max_side(sf.image))

            captions = generate_captions_for_video(output_dir, selected, scenes, styles)

            # Guarantee every requested style has a non-empty caption.
            for style in styles:
                if not captions.get(style):
                    captions[style] = (
                        f"A {sum(s.duration for s in scenes):.0f}-second video with "
                        f"{len(scenes)} scene(s); detailed captioning was unavailable."
                    )

            logger.info(f"Task {task_id} completed in {time.time() - task_start:.1f}s")
            return task_id, captions
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return task_id, {
                style: "Processing failed for this video clip." for style in styles
            }
        finally:
            if video_path:
                try:
                    os.unlink(video_path)
                except OSError:
                    pass

    # Process tasks concurrently: downloads/decodes of one task overlap the
    # API calls of another. Worker count is env-tunable for small judge VMs.
    workers = min(MAX_TASK_WORKERS, max(1, len(tasks)))
    logger.info(f"Processing {len(tasks)} tasks with {workers} workers "
                f"(deadline {DEADLINE_SECONDS:.0f}s)")
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(_process_task, task) for task in tasks]
    pending = set(futures)
    try:
        for future in as_completed(futures, timeout=max(5.0, deadline - time.time())):
            pending.discard(future)
            task_id, captions = future.result()
            results_by_id[task_id] = captions
            if time.time() > deadline:
                logger.warning("Deadline reached; writing results with completed tasks")
                break
    except FuturesTimeoutError:
        logger.warning(
            f"Deadline ({DEADLINE_SECONDS:.0f}s) hit with {len(pending)} task(s) "
            f"unfinished; writing partial results with placeholders"
        )

    # Write output (placeholders remain for anything unfinished).
    results = [{"task_id": tid, "captions": results_by_id[tid]} for tid in task_order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    done = sum(1 for f in futures if f.done())
    logger.info(f"{done}/{len(tasks)} tasks completed in {total_time:.1f}s")
    logger.info(f"Results written to {output_path}")
    # Force-exit: worker threads may still be blocked on network I/O after a
    # deadline break, and results are already safely on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main(args):
    video_path = args.video_path
    if not video_path or not os.path.isfile(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Determine which styles to generate
    if args.all_styles:
        styles = REPORT_STYLES
    else:
        styles = [args.style]

    run_start = time.time()
    logger.info(f"=== START pipeline for: {video_path} ===")

    scenes, captured = detect_scenes(video_path)
    logger.info(f"Number of scenes found: {len(scenes)}")

    output_dir = Path("output") / _video_short_name(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_dir.resolve()}")

    # --- Embedding-based keyframe selection (replaces single-frame extraction) ---
    selected = select_keyframes(video_path, scenes, captured)
    logger.info(f"✅ Selected keyframes: {len(selected)} (MAX_FRAMES budget enforced globally)")

    keyframes_data = []
    save_start = time.time()
    for i, sf in enumerate(selected):
        filename = f"keyframe_{i:03d}.jpg"
        filepath = output_dir / filename
        # image is a BGR numpy array from cv2; downscale longest side, then write JPEG.
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
    with open(keyframes_json_path, "w", encoding="utf-8") as f:
        json.dump(keyframes_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved: keyframes.json ({len(keyframes_data)} frames)")

    # --- Preserve scene-detection metadata for the downstream handoff ---
    scenes_data = []
    for scene in scenes:
        scenes_data.append({
            "scene_number": scene.scene_number,
            "start_time": format_timecode(scene.start_time),
            "end_time": format_timecode(scene.end_time),
            "duration": scene.duration,
        })
    scenes_json_path = output_dir / "scenes.json"
    with open(scenes_json_path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: scenes.json ({len(scenes_data)} scenes)")

    total = time.time() - run_start
    logger.info("✅ === PHASE ARTIFACT COMPLETE ===")
    logger.info(
        f"✅ VERIFY: scenes={len(scenes)} | keyframes={len(selected)} "
        f"(<=MAX_FRAMES={MAX_FRAMES}) | total runtime={total:.2f}s"
    )
    logger.info(f"Output: {output_dir.resolve()}")

    # --- Report Generation (if enabled) ---
    if args.reports:
        report_start = time.time()
        captions = generate_reports_pipeline(
            output_dir=output_dir,
            selected=selected,
            scenes=scenes,
            styles=styles,
        )
        report_total = time.time() - report_start
        logger.info(f"Report generation completed in {report_total:.2f}s")

        # Save captions to JSON
        output_json = {
            "video": video_path,
            "scenes": scenes_data,
            "keyframes": keyframes_data,
            "captions": captions
        }
        json_path = output_dir / "captions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Captions saved to: {json_path.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    if args.input_json:
        # Batch mode: read tasks from JSON, write results
        competition_main(args.input_json, args.output_json)
    elif Path("/input/tasks.json").exists():
        # Docker competition mode
        competition_main()
    elif args.video_path:
        # Single video mode
        main(args)
    else:
        print("Error: Provide --input <tasks.json> or a video file path")
        sys.exit(1)
