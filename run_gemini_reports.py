"""Gemini vision + Groq text: 2 frames in parallel."""

import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from config import GEMINI_VISION_MODEL, GROQ_TEXT_MODEL
from services.gemini_client import get_gemini_client
from services.groq_client import get_groq_client

FRAMES_DIR = Path(r"C:\Users\a\Downloads\Video-Caption-temp1\Video-Caption-temp1\frames")
OUTPUT_DIR = Path(r"C:\Users\a\Downloads\Video-Caption-temp1\Video-Caption-temp1\output")

VISION_PROMPT = "Describe what you see in this frame. Be brief."

TEXT_PROMPT = """Given this scene description, write captions in 4 styles. Return ONLY valid JSON, no markdown:
{{"formal": "...", "sarcastic": "...", "humorous_tech": "...", "humorous_non_tech": "..."}}

Scene: {desc}"""


def process_frame(gemini, groq, frame, index):
    # Vision: Gemini
    t0 = time.perf_counter()
    desc = gemini.analyze_images_batch(
        image_paths=[str(frame)],
        prompt=VISION_PROMPT,
        model=GEMINI_VISION_MODEL,
        max_tokens=256,
    )
    vision_time = time.perf_counter() - t0
    desc = desc.strip()

    # Text: Groq
    t1 = time.perf_counter()
    raw = groq.generate_text(
        prompt=TEXT_PROMPT.format(desc=desc),
        model=GROQ_TEXT_MODEL,
        max_tokens=1024,
    )
    text_time = time.perf_counter() - t1
    raw = raw.strip()

    # Parse JSON
    cleaned = re.sub(r"```json\s*", "", raw).strip().strip("`")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        caps = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        caps = {}

    return {
        "index": index, "name": frame.name, "desc": desc,
        "captions": caps, "vision_time": vision_time, "text_time": text_time,
    }


def main():
    frames = sorted(FRAMES_DIR.glob("keyframe_*.jpg"))
    print(f"Using {len(frames)} frames: {[f.name for f in frames]}\n")

    gemini = get_gemini_client()
    groq = get_groq_client()
    total_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process_frame, gemini, groq, f, i): i for i, f in enumerate(frames)}
        results = {}
        for future in as_completed(futures):
            r = future.result()
            results[r["index"]] = r
            print(f"Frame {r['index']} ({r['name']}): vision={r['vision_time']:.2f}s groq={r['text_time']:.2f}s")
            print(f"  desc: {r['desc'][:80]}")
            for s, c in r["captions"].items():
                print(f"  {s}: {c[:60]}")
            print()

    total_time = time.perf_counter() - total_start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_captions = results[max(results.keys())]["captions"] if results else {}
    result = [{"task_id": "v1", "captions": final_captions}]
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Total: {total_time:.2f}s (parallel)")
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
