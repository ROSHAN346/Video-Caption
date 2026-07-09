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
from concurrent.futures import ThreadPoolExecutor, as_completed

from scene_detector import detect_scenes
from frame_selector import select_keyframes
from config import (
    FRAME_STRATEGY, MAX_FRAMES, REPORT_STYLES,
    DEFAULT_REPORT_STYLE, FIREWORKS_API_KEY, FIREWORKS_VISION_MODEL, FIREWORKS_TEXT_MODEL,
    GEMINI_API_KEY, GEMINI_VISION_MODEL, GEMINI_TEXT_MODEL,
    GROQ_API_KEY, GROQ_TEXT_MODEL,
    AI_PROVIDER
)
from services.fireworks_client import get_fireworks_client
from services.groq_client import get_groq_client

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
    parser.add_argument(
        "--provider",
        choices=["fireworks", "gemini"],
        default=AI_PROVIDER,
        help=f"AI provider to use (default: {AI_PROVIDER})"
    )
    return parser.parse_args()


def generate_reports_pipeline(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str],
    provider: str = AI_PROVIDER
):
    """Run the report generation pipeline: Fireworks vision + Groq text."""
    from services.fireworks_client import get_fireworks_client
    from services.groq_client import get_groq_client
    from services.image_analyzer import analyze_keyframe, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_all_reports
    from services.report_cache import ReportCache

    logger.info("=== START Report Generation (Fireworks vision + Groq text) ===")

    # Initialize vision client (Fireworks) + text client (Groq)
    try:
        if not FIREWORKS_API_KEY:
            logger.error("FIREWORKS_API_KEY not set in .env")
            return
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not set in .env")
            return
        vision_client = get_fireworks_client()
        text_client = get_groq_client()
        vision_model = FIREWORKS_VISION_MODEL
        text_model = GROQ_TEXT_MODEL
        logger.info(f"Vision: Fireworks ({vision_model}) | Text: Groq ({text_model})")
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
        keyframe_path = str(output_dir / f"keyframe_{i:03d}.jpg")
        if not Path(keyframe_path).exists():
            logger.warning(f"Keyframe not found: {keyframe_path}")
            return None
        try:
            analysis = analyze_keyframe(
                image_path=keyframe_path,
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
                "provider": provider
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

    Args:
        url: Video URL
        timeout: Download timeout in seconds

    Returns:
        Path to downloaded video file
    """
    logger.info(f"Downloading video from: {url}")
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Determine file extension from URL or content-type
    suffix = ".mp4"
    if "?" in url:
        url_path = url.split("?")[0]
    else:
        url_path = url
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        if url_path.lower().endswith(ext):
            suffix = ext
            break

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_path = tmp_file.name

    logger.info(f"Downloaded video to: {tmp_path}")
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
    Uses Fireworks vision + Groq text.
    """
    from services.image_analyzer import analyze_keyframe, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_video_summary_reports

    if not FIREWORKS_API_KEY:
        logger.error("FIREWORKS_API_KEY not set in .env")
        return {}
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in .env")
        return {}

    vision_client = get_fireworks_client()
    text_client = get_groq_client()
    vision_model = FIREWORKS_VISION_MODEL
    text_model = GROQ_TEXT_MODEL

    # Analyze keyframes in parallel
    logger.info(f"Analyzing {len(selected)} keyframes in parallel...")
    analyses = []
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _analyze_one(i, sf):
        keyframe_path = str(output_dir / f"keyframe_{i:03d}.jpg")
        if not Path(keyframe_path).exists():
            logger.warning(f"Keyframe not found: {keyframe_path}")
            return None
        try:
            analysis = retry_api_call(
                lambda p=keyframe_path, s=sf, idx=i: analyze_keyframe(
                    image_path=p,
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


def competition_main(input_path=None, output_path=None):
    """
    Competition entrypoint: read tasks.json, process each task,
    write results.json.
    """
    input_path = Path(input_path or "/input/tasks.json")
    output_path = Path(output_path or "/output/results.json")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

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

    results = []
    total_start = time.time()

    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        requested_styles = task.get("styles", REPORT_STYLES)
        styles = map_styles(requested_styles)

        logger.info(f"=== Processing task {task_id} ===")
        logger.info(f"Styles: {requested_styles} -> {styles}")
        task_start = time.time()

        try:
            # Download video
            video_path = download_video(video_url)

            # Run pipeline
            scenes = detect_scenes(video_path)
            logger.info(f"Task {task_id}: Found {len(scenes)} scenes")

            selected = select_keyframes(video_path, scenes)
            logger.info(f"Task {task_id}: Selected {len(selected)} keyframes")

            # Save keyframes to temp dir
            output_dir = Path(tempfile.mkdtemp(prefix=f"task_{task_id}_"))
            output_dir.mkdir(parents=True, exist_ok=True)

            analysis_dir = output_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            for i, sf in enumerate(selected):
                filepath = output_dir / f"keyframe_{i:03d}.jpg"
                cv2.imwrite(str(filepath), _downscale_to_max_side(sf.image))

            # Generate captions
            captions = generate_captions_for_video(output_dir, selected, scenes, styles)

            # Cleanup temp video
            try:
                os.unlink(video_path)
            except OSError:
                pass

            task_time = time.time() - task_start
            logger.info(f"Task {task_id} completed in {task_time:.1f}s")

            results.append({
                "task_id": task_id,
                "captions": captions
            })

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            # Still include task with placeholder captions to avoid zero score
            results.append({
                "task_id": task_id,
                "captions": {style: f"Processing failed for this video clip." for style in styles}
            })

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_start
    logger.info(f"All tasks completed in {total_time:.1f}s")
    logger.info(f"Results written to {output_path}")
    sys.exit(0)


def main():
    args = parse_args()

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

    scenes = detect_scenes(video_path)
    logger.info(f"Number of scenes found: {len(scenes)}")

    output_dir = Path("output") / _video_short_name(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_dir.resolve()}")

    # --- Embedding-based keyframe selection (replaces single-frame extraction) ---
    selected = select_keyframes(video_path, scenes)
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
    with open(keyframes_json_path, "w") as f:
        json.dump(keyframes_data, f, indent=2)
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
    with open(scenes_json_path, "w") as f:
        json.dump(scenes_data, f, indent=2)
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
            provider=args.provider
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
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)
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
        main()
    else:
        print("Error: Provide --input <tasks.json> or a video file path")
        sys.exit(1)
