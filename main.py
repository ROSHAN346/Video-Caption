import sys
import json
import os
import time
import re
import logging
import argparse
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
    parser.add_argument("video_path", help="Path to video file")
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
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation"
    )
    parser.add_argument(
        "--provider",
        choices=["huggingface", "fireworks"],
        default=AI_PROVIDER,
        help=f"AI provider to use (default: {AI_PROVIDER})"
    )
    return parser.parse_args()


def generate_reports_pipeline(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str],
    generate_pdf: bool = True,
    provider: str = AI_PROVIDER
):
    """Run the report generation pipeline."""
    from services.hf_client import get_hf_client
    from services.fireworks_client import get_fireworks_client
    from services.image_analyzer import analyze_keyframe, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_all_reports
    from services.report_cache import ReportCache
    from services.pdf_generator import generate_report_pdf

    logger.info(f"=== START Report Generation (provider: {provider}) ===")

    # Initialize client based on provider
    try:
        if provider == "fireworks":
            if not FIREWORKS_API_KEY:
                logger.error(
                    "FIREWORKS_API_KEY not set. Please create a .env file with your key.\n"
                    "Get yours at: https://fireworks.ai/account/api-keys"
                )
                return
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
                return
            client = get_hf_client()
            vision_model = HF_VISION_MODEL
            text_model = HF_TEXT_MODEL
            logger.info("Using Hugging Face provider")
    except Exception as e:
        logger.error(f"Failed to initialize {provider} client: {e}")
        return

    cache = ReportCache(str(output_dir / "reports"))

    # Create analysis directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Analyze keyframes
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
                frame_index=i
            )
            # Skip if analysis failed (empty/unavailable)
            if analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
                logger.warning(f"Frame {i} analysis failed, skipping")
                continue
            analyses.append(analysis)

            # Save individual analysis
            analysis_path = analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
            save_analysis(analysis, str(analysis_path))

            logger.info(f"Analyzed frame {i}: {analysis.get('summary', 'N/A')[:50]}...")
        except Exception as e:
            logger.warning(f"Frame {i} failed: {e}, skipping")
            continue

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
    for scene_id, scene_data in scene_analyses.items():
        logger.info(f"Processing scene {scene_id}...")

        reports = generate_all_reports(scene_data, client, styles, text_model)

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

            # Generate PDF if enabled
            if generate_pdf:
                keyframe_path = str(output_dir / f"keyframe_000.jpg")
                pdf_path = cache.get_pdf_path(scene_id, style)
                try:
                    generate_report_pdf(
                        scene_id=scene_id,
                        style=style,
                        report=report,
                        keyframe_path=keyframe_path,
                        scene_data=scene_data,
                        output_path=str(pdf_path)
                    )
                    logger.info(f"Generated PDF: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error generating PDF for {style}: {e}")

    # Print summary
    stats = cache.get_cache_stats()
    logger.info("=== Report Generation Complete ===")
    logger.info(f"Scenes processed: {stats['total_scenes']}")
    logger.info(f"Reports generated: {stats['total_reports']}")
    logger.info(f"Output directory: {output_dir / 'reports'}")


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


def main():
    args = parse_args()

    video_path = args.video_path
    if not os.path.isfile(video_path):
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
        generate_reports_pipeline(
            output_dir=output_dir,
            selected=selected,
            scenes=scenes,
            styles=styles,
            generate_pdf=not args.no_pdf,
            provider=args.provider
        )
        report_total = time.time() - report_start
        logger.info(f"Report generation completed in {report_total:.2f}s")


if __name__ == "__main__":
    main()
