import os
import sys
import re
import json
import time
import math
import base64
import requests
import tempfile
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# External library for scene detection
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def download_video(url: str) -> str:
    print(f"[Download] Fetching {url}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return path

def extract_and_compress_frames(video_path: str):
    print(f"[Extract] Processing {video_path}")
    
    # 1. Get 10 uniform frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    uniform_indices = []
    if total_frames > 0:
        uniform_indices = [int(i) for i in np.linspace(0, total_frames - 1, 8)]
        
    cap.release()
    
    # 2. Get 1 frame per scene
    scene_indices = []
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        for scene in scenes:
            # take middle frame of the scene
            mid_frame = (scene[0].get_frames() + scene[1].get_frames()) // 2
            scene_indices.append(mid_frame)
    except Exception as e:
        print(f"[Extract] Scene detection failed: {e}")
        
    # Combine and sort unique indices
    all_indices = sorted(list(set(uniform_indices + scene_indices)))[:10]
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Compress: resize to max 720p to save bandwidth
            h, w = frame.shape[:2]
            max_dim = 512
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
            timestamp = idx / fps
            frames_data.append({
                "idx": idx,
                "timestamp": timestamp,
                "frame": frame
            })
            
    cap.release()
    
    # 3. Delete redundant frames (simple pixel MSE threshold)
    unique_frames = []
    for data in frames_data:
        if not unique_frames:
            unique_frames.append(data)
            continue
            
        prev_frame = unique_frames[-1]["frame"]
        curr_frame = data["frame"]
        
        # Only compare if sizes match
        if prev_frame.shape == curr_frame.shape:
            mse = np.mean((prev_frame.astype(np.float32) - curr_frame.astype(np.float32)) ** 2)
            if mse < 100:  # Highly similar
                continue
                
        unique_frames.append(data)
        
    print(f"[Extract] Kept {len(unique_frames)} non-redundant frames out of {len(frames_data)}")
    
    # Convert frames to base64
    for data in unique_frames:
        _, buffer = cv2.imencode('.jpg', data["frame"], [cv2.IMWRITE_JPEG_QUALITY, 50])
        data["b64"] = base64.b64encode(buffer).decode('utf-8')
        del data["frame"] # free memory
        
    return unique_frames

def analyze_batch_vision(batch, batch_index):
    if not FIREWORKS_API_KEY:
        return f"Batch {batch_index}: No API Key"
        
    print(f"[Vision] Sending batch {batch_index} ({len(batch)} frames)")
    
    content = [{"type": "text", "text": "Describe the events, objects, and actions in these frames. Pay attention to the chronological sequence."}]
    
    for item in batch:
        content.append({
            "type": "text",
            "text": f"Timestamp: {item['timestamp']:.2f}s"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{item['b64']}"
            }
        })
        
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "accounts/fireworks/models/minimax-m3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 600
    }
    
    try:
        resp = requests.post("https://api.fireworks.ai/inference/v1/chat/completions", headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        try:
            msg = data["choices"][0]["message"]
            # minimax-m3 is a reasoning model, so it might output 'reasoning_content' before 'content'.
            # If it hits max_tokens early, 'content' might be missing entirely.
            text = msg.get("content")
            if not text:
                text = msg.get("reasoning_content", "")
            return text
        except KeyError as e:
            print(f"[Vision Error] Batch {batch_index}: Missing key {e}. Raw response: {json.dumps(data)}")
            return ""
    except Exception as e:
        print(f"[Vision Error] Batch {batch_index}: {e}")
        if 'resp' in locals() and hasattr(resp, 'text'):
            print(f"Raw HTTP Response: {resp.text}")
        return ""

TONE_PROMPTS = {
    "formal": (
        "You are a professional visual documentation specialist. "
        "You write precise, objective, third-person captions for video footage. "
        "You never use humor, slang, or subjective language. "
        "You never reference AI, models, or analysis tools. "
        "Output ONLY the final 2-sentence caption (25-60 words). "
        "No reasoning, no explanation, no chain-of-thought, no labels, no headers."
    ),
    "sarcastic": (
        "You are a dry, sardonic commentator who finds specific situations amusing. "
        "Your sarcasm is grounded in concrete details — never generic. "
        "You use understatement, false praise, or deadpan observation — one technique per caption. "
        "You never reference AI, models, or analysis tools. "
        "Output ONLY the final 2-sentence caption (25-60 words). "
        "No reasoning, no explanation, no chain-of-thought, no labels, no headers."
    ),
    "humorous_tech": (
        "You are a senior software engineer who instinctively maps real-world situations "
        "to programming concepts, error codes, design patterns, and developer culture. "
        "Your humor is clever and specific — the tech metaphor must fit the actual scene. "
        "You never reference AI, models, or analysis tools. "
        "Output ONLY the final 2-sentence caption (25-60 words). "
        "No reasoning, no explanation, no chain-of-thought, no labels, no headers."
    ),
    "humorous_non_tech": (
        "You are a warm, observational comedian who finds everyday life hilarious. "
        "Your humor is relatable — the kind of thing someone texts a friend. "
        "No tech jargon, no internet memes, no forced punchlines. Ground every joke in a specific detail. "
        "You never reference AI, models, or analysis tools. "
        "Output ONLY the final 2-sentence caption (25-60 words). "
        "No reasoning, no explanation, no chain-of-thought, no labels, no headers."
    )
}

def generate_tones(vision_summary):
    if not GROQ_API_KEY:
        return {"error": "No GROQ_API_KEY"}

    print("[LLM] Generating 4 tones in a single request...")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are a versatile caption writer. Given a video analysis, generate exactly 4 captions "
        "in different tones. Output ONLY valid JSON with these keys: "
        "formal, sarcastic, humorous_tech, humorous_non_tech. "
        "Each caption must be 2-3 sentences (25-60 words), standalone, and explicitly introduce "
        "the main subject. No reasoning, no labels outside JSON."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Video analysis:\n{vision_summary}\n\n"
             "Return JSON with keys: formal, sarcastic, humorous_tech, humorous_non_tech"}
        ],
        "max_tokens": 700,
        "temperature": 0.0
    }

    try:
        for _attempt in range(2):
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"[LLM] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            break
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        if not content:
            print(f"[LLM Error] empty content. Raw: {resp.text}")
            return {"error": "empty response from model"}
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        return json.loads(content)
    except Exception as e:
        print(f"[LLM Error] {e}")
        return {"error": str(e)}

def process_video(task):
    task_id = task.get("task_id", "unknown")
    url = task.get("video_url")
    start = time.time()

    try:
        # 1. Download
        video_path = download_video(url)

        # 2. Extract, compress, delete redundant
        frames = extract_and_compress_frames(video_path)
        os.unlink(video_path)

        # 3. Send to vision in parallel batches of 2
        batches = [frames[i:i+2] for i in range(0, len(frames), 2)]
        vision_results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_batch_vision, batch, idx) for idx, batch in enumerate(batches)]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    vision_results.append(res)

        combined_vision_summary = "\n\n".join(vision_results)

        # 4. Generate 4 tones
        tones = generate_tones(combined_vision_summary)

        elapsed = round(time.time() - start, 2)
        print(f"[Timing] Task {task_id}: {elapsed}s")
        return {
            "task_id": task_id,
            "captions": tones
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[Timing] Task {task_id}: {elapsed}s (failed)")
        print(f"[Error] Task {task_id}: {e}")
        return {
            "task_id": task_id,
            "error": str(e)
        }

def main():
    # Validate API keys at startup — fail fast with clear error
    if not FIREWORKS_API_KEY:
        print("[FATAL] FIREWORKS_API_KEY is not set. Cannot continue.")
        sys.exit(1)
    if not GROQ_API_KEY:
        print("[FATAL] GROQ_API_KEY is not set. Cannot continue.")
        sys.exit(1)

    # Spec-mandated paths: /input/tasks.json → /output/results.json
    input_file = "/input/tasks.json" if os.path.exists("/input/tasks.json") else "tasks.json"
    output_file = "/output/results.json"

    if not os.path.exists(input_file):
        print(f"[FATAL] Input file '{input_file}' not found.")
        sys.exit(1)

    with open(input_file, "r") as f:
        tasks = json.load(f)

    results = []

    print(f"Starting parallel processing for {len(tasks)} tasks (max 3 at a time)...")

    # Process up to 3 videos in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {executor.submit(process_video, task): task for task in tasks}

        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            print(f"✅ Finished task: {res.get('task_id')}")

    # Ensure output directory exists (Docker mounts it but it must exist)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Results written to {output_file}")
    sys.exit(0)

if __name__ == "__main__":
    main()
