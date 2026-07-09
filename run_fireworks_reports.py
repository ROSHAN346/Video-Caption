import io
import json
import re
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import FIREWORKS_VISION_MODEL, GROQ_TEXT_MODEL
from services.fireworks_client import get_fireworks_client
from services.groq_client import get_groq_client
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VISION_PROMPT = "Describe what you see in this frame. Be brief."

TEXT_PROMPT = """These are descriptions from multiple frames of the same video.
Write captions summarizing the entire video in 4 styles.
Return ONLY valid JSON, no markdown:
{{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}

Frame descriptions:
{descriptions}"""


def describe_frame(fireworks, image_path, model):
    start = time.time()
    try:
        MAX_SIZE = 768
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            image_bytes = buf.getvalue()

        desc = fireworks.analyze_image_base64(
            image_bytes=image_bytes,
            prompt=VISION_PROMPT,
            model=model,
            mime_type="jpeg",
            max_tokens=256,
        ).strip()
        elapsed = time.time() - start
        logger.info(f"[{image_path.name}] Vision {elapsed:.2f}s -> {desc[:80]}")
        return image_path.name, desc, None
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[{image_path.name}] Failed {elapsed:.2f}s: {e}")
        return image_path.name, None, str(e)


def main():
    frames_dir = Path(r"C:\Users\a\Downloads\Video-Caption-temp1\Video-Caption-temp1\frames")
    output_dir = Path(r"C:\Users\a\Downloads\Video-Caption-temp1\Video-Caption-temp1\output")

    if not frames_dir.exists():
        logger.error(f"Directory '{frames_dir.resolve()}' does not exist.")
        return

    image_paths = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpeg"))
    )

    if not image_paths:
        logger.warning(f"No images found in '{frames_dir.resolve()}'.")
        return

    logger.info(f"Found {len(image_paths)} frames to process.")

    fireworks = get_fireworks_client()
    groq = get_groq_client()
    total_start = time.time()

    # Phase 1: Parallel vision (Fireworks)
    descriptions = []
    with ThreadPoolExecutor(max_workers=min(len(image_paths), 10)) as executor:
        futures = {executor.submit(describe_frame, fireworks, path, FIREWORKS_VISION_MODEL): path for path in image_paths}
        for future in as_completed(futures):
            name, desc, error = future.result()
            if desc:
                descriptions.append(f"Frame {name}: {desc}")

    if not descriptions:
        logger.error("No frames were successfully described.")
        return

    # Phase 2: Single Groq call
    combined = "\n".join(descriptions)
    prompt = TEXT_PROMPT.format(descriptions=combined)

    logger.info("Groq text call (1 call, all frames combined)...")
    t0 = time.time()
    raw = groq.generate_text(prompt=prompt, model=GROQ_TEXT_MODEL, max_tokens=1024).strip()
    groq_time = time.time() - t0
    logger.info(f"Groq {groq_time:.2f}s -> {raw[:200]}")

    # Parse JSON
    cleaned = re.sub(r"```json\s*", "", raw).strip().strip("`")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        captions = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        captions = {}
        logger.warning("JSON parse failed")

    total_elapsed = time.time() - total_start

    # Write results_fireworks.json
    output_dir.mkdir(parents=True, exist_ok=True)
    result = [{"task_id": "v1", "captions": captions}]
    out_path = output_dir / "results_fireworks.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(f"=== COMPLETE ===")
    logger.info(f"Total: {total_elapsed:.2f}s ({len(image_paths)} vision + 1 groq)")
    logger.info(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
