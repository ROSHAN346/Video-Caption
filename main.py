import sys
import json
import os
import time
import logging
import argparse
import tempfile
import requests
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from scene_detector import detect_scenes
from frame_selector import select_keyframes
from services.proxy_stream import ProxyStream
from config import (
    REPORT_STYLES, FIREWORKS_API_KEY, FIREWORKS_VISION_MODEL,
    GROQ_API_KEY, GROQ_TEXT_MODEL
)
from services.fireworks_client import get_fireworks_client
from services.groq_client import get_groq_client

MAX_SAVE_SIDE = 1280

def _downscale_to_max_side(frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= MAX_SAVE_SIDE:
        return frame
    scale = MAX_SAVE_SIDE / float(long)
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Keyframe Extraction with AI Report Generation"
    )
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
    return parser.parse_args()




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
    status_codes = []
    for attempt in range(max_retries):
        try:
            res = func()
            status_codes.append(200)
            return res, status_codes
        except Exception as e:
            code = 500
            if hasattr(e, "status_code"):
                code = e.status_code
            elif "429" in str(e) or "rate" in str(e).lower():
                code = 429
            status_codes.append(code)
            
            if code == 429:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                logger.error(f"API call failed: {e}")
                return None, status_codes
    logger.error(f"API call failed after {max_retries} retries")
    return None, status_codes


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
    from services.image_analyzer import analyze_keyframes_causal
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

    # Gather keyframe paths
    image_paths = []
    for i, sf in enumerate(selected):
        keyframe_path = output_dir / f"keyframe_{i:03d}.jpg"
        if keyframe_path.exists():
            image_paths.append(str(keyframe_path))
            
    if not image_paths:
        logger.error("No keyframes found to analyze.")
        return {style: "No content generated." for style in styles}

    # 1. Single Multi-Image API call to analyze the whole video chronologically
    logger.info(f"Analyzing causal sequence of {len(image_paths)} keyframes...")
    
    vlm_start = time.time()
    analysis, vlm_codes = retry_api_call(
        lambda: analyze_keyframes_causal(
            image_paths=image_paths,
            client=vision_client,
            model=vision_model
        )
    )
    vlm_time = time.time() - vlm_start

    if not analysis or analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
        logger.warning("Multi-image API analysis failed. Using fallback.")
        activities = ", ".join([f"Scene {s.scene_number}: {s.duration:.1f}s segment" for s in scenes])
        scene_analyses = {
            "all": {
                "activities": activities,
                "summary": activities
            }
        }
    else:
        # 2. Package it into the scene_analyses format expected by the report generator
        scene_analyses = {
            "all": analysis
        }

    # 3. Generate video-level captions using Groq
    logger.info(f"Generating {len(styles)} caption styles...")
    llm_start = time.time()
    captions, llm_codes = retry_api_call(
        lambda: generate_video_summary_reports(scene_analyses, text_client, styles, text_model)
    )
    llm_time = time.time() - llm_start

    # Guarantee all styles have captions
    if not captions:
        captions = {}
    for style in styles:
        if style not in captions or not captions[style]:
            captions[style] = f"Video content analyzed: {', '.join(s.get('activities', 'unknown') for s in scene_analyses.values())}"

    return captions, {
        "vlm_time": vlm_time,
        "vlm_codes": vlm_codes,
        "llm_time": llm_time,
        "llm_codes": llm_codes
    }


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
    stats_list = []
    frame_stats_list = []
    total_start = time.time()

    def process_task(task):
        task_id = task["task_id"]
        video_url = task["video_url"]
        requested_styles = task.get("styles", REPORT_STYLES)
        styles = map_styles(requested_styles)
        
        logger.info(f"=== Processing task {task_id} ===")
        task_start = time.time()
        
        try:
            # Download
            t0 = time.time()
            video_path = download_video(video_url)
            download_time = time.time() - t0
            
            # Scene Detect
            t0 = time.time()
            proxy = ProxyStream(video_path, proxy_side=360)
            scenes = detect_scenes(proxy)
            scene_time = time.time() - t0
            
            # Select
            t0 = time.time()
            selected, f_stats = select_keyframes(proxy, scenes)
            frame_time = time.time() - t0

            # Generate captions
            output_dir = Path(tempfile.mkdtemp(prefix=f"task_{task_id}_"))
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, sf in enumerate(selected):
                filepath = output_dir / f"keyframe_{i:03d}.jpg"
                cv2.imwrite(str(filepath), _downscale_to_max_side(sf.image))

            captions, api_stats = generate_captions_for_video(output_dir, selected, scenes, styles)

            # Cleanup
            try:
                os.unlink(video_path)
            except OSError:
                pass

            task_time = time.time() - task_start

            res_dict = {
                "task_id": task_id,
                "captions": captions
            }
            
            stat_dict = {
                "Task": task_id,
                "DL(s)": f"{download_time:.1f}",
                "Scene(s)": f"{scene_time:.1f}",
                "Sel(s)": f"{frame_time:.1f}",
                "VLM(s)": f"{api_stats['vlm_time']:.1f}",
                "VLM_Codes": str(api_stats['vlm_codes']),
                "LLM(s)": f"{api_stats['llm_time']:.1f}",
                "LLM_Codes": str(api_stats['llm_codes']),
                "Total(s)": f"{task_time:.1f}"
            }
            
            f_stat_dict = {
                "Task": task_id,
                "Candidates": f_stats["candidates"],
                "Read": f_stats["read"],
                "Pruned(read)": f_stats["pruned_read"],
                "Pruned(sim)": f_stats["pruned_sim"],
                "Selected": f_stats["selected"]
            }
            return res_dict, stat_dict, f_stat_dict

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            res_dict = {
                "task_id": task_id,
                "captions": {style: f"Processing failed for this video clip." for style in styles}
            }
            stat_dict = {
                "Task": task_id,
                "DL(s)": "-", "Scene(s)": "-", "Sel(s)": "-", 
                "VLM(s)": "-", "VLM_Codes": "-", 
                "LLM(s)": "-", "LLM_Codes": "-",
                "Total(s)": "ERROR"
            }
            f_stat_dict = {
                "Task": task_id,
                "Candidates": "-", "Read": "-", "Pruned(read)": "-", "Pruned(sim)": "-", "Selected": "-"
            }
            return res_dict, stat_dict, f_stat_dict

    # Run all tasks concurrently
    with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        
        for future in as_completed(futures):
            res_dict, stat_dict, f_stat_dict = future.result()
            results.append(res_dict)
            stats_list.append(stat_dict)
            frame_stats_list.append(f_stat_dict)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print nice tables
    print("\n" + "="*95)
    print(" TIMING & RESPONSE CODE STATISTICS ".center(95, "="))
    print("="*95)
    print(f"{'Task':<20} | {'DL(s)':<6} | {'Scene(s)':<8} | {'Sel(s)':<6} | {'VLM(s)':<6} | {'VLM_Codes':<10} | {'LLM(s)':<6} | {'LLM_Codes':<10} | {'Total(s)':<8}")
    print("-" * 95)
    for s in stats_list:
        print(f"{s['Task']:<20} | {s['DL(s)']:<6} | {s['Scene(s)']:<8} | {s['Sel(s)']:<6} | {s['VLM(s)']:<6} | {s['VLM_Codes']:<10} | {s['LLM(s)']:<6} | {s['LLM_Codes']:<10} | {s['Total(s)']:<8}")
    print("="*95 + "\n")

    print("\n" + "="*80)
    print(" FRAME FILTERING & SELECTION STATISTICS ".center(80, "="))
    print("="*80)
    print(f"{'Task':<20} | {'Candidates':<10} | {'Read':<6} | {'Pruned (I/O)':<12} | {'Pruned (Sim)':<12} | {'Selected':<8}")
    print("-" * 80)
    for fs in frame_stats_list:
        print(f"{fs['Task']:<20} | {fs['Candidates']:<10} | {fs['Read']:<6} | {fs['Pruned(read)']:<12} | {fs['Pruned(sim)']:<12} | {fs['Selected']:<8}")
    print("="*80 + "\n")

    print(f"Results written to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    if args.input_json and args.output_json:
        competition_main(args.input_json, args.output_json)
    elif Path("/input/tasks.json").exists():
        competition_main()
    else:
        logger.error("Error: Provide --input <tasks.json> and --output <results.json> or run in Docker environment.")
        sys.exit(1)

