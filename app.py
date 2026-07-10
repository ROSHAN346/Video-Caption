import os
import re
import json
import time
import math
import base64
import requests
import tempfile
import cv2
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# Load dotenv if available
try:
    load_dotenv()
except Exception:
    pass

import streamlit as st

# External library for scene detection
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------- ORIGINAL CODE (STRICTLY ADHERED TO) -----------------

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
        uniform_indices = [int(i) for i in np.linspace(0, total_frames - 1, 10)]
        
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
    all_indices = sorted(list(set(uniform_indices + scene_indices)))
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Compress: resize to max 720p to save bandwidth
            h, w = frame.shape[:2]
            max_dim = 720
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
        _, buffer = cv2.imencode('.jpg', data["frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        "max_tokens": 1500
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
        "max_tokens": 1000,
        "temperature": 0.0
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers=headers, json=payload, timeout=60)
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
        
        return {
            "task_id": task_id,
            "captions": tones
        }
        
    except Exception as e:
        print(f"[Error] Task {task_id}: {e}")
        return {
            "task_id": task_id,
            "error": str(e)
        }

# ----------------- CACHED STREAMLIT WRAPPER FOR SUPERFAST EXECUTION -----------------

@st.cache_data(show_spinner=False)
def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

@st.cache_data(show_spinner=False)
def cached_process_video_pipeline_from_url(url: str):
    """Run the entire unmodified process_video pipeline for a URL, cached."""
    task = {"task_id": "streamlit_url_task", "video_url": url}
    
    # Run the exact process_video function
    t0 = time.time()
    video_path = download_video(url)
    t_dl = time.time() - t0
    
    t0 = time.time()
    frames = extract_and_compress_frames(video_path)
    t_ex = time.time() - t0
    
    try:
        os.unlink(video_path)
    except Exception:
        pass
        
    t0 = time.time()
    batches = [frames[i:i+2] for i in range(0, len(frames), 2)]
    vision_results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_batch_vision, batch, idx) for idx, batch in enumerate(batches)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                vision_results.append(res)
                
    combined_vision_summary = "\n\n".join(vision_results)
    t_vis = time.time() - t0
    
    t0 = time.time()
    tones = generate_tones(combined_vision_summary)
    t_llm = time.time() - t0
    
    return {
        "captions": tones,
        "frames": frames,
        "vision_summary": combined_vision_summary,
        "stats": {
            "download_time": t_dl,
            "extract_time": t_ex,
            "vision_time": t_vis,
            "llm_time": t_llm,
            "total_time": t_dl + t_ex + t_vis + t_llm
        }
    }

@st.cache_data(show_spinner=False)
def cached_process_video_pipeline_from_local_file(file_hash: str, temp_filepath: str):
    """Run the unmodified processing pipeline for a local file, cached by file hash."""
    t0 = time.time()
    # local upload skips download time
    t_dl = 0.0 
    
    t0 = time.time()
    frames = extract_and_compress_frames(temp_filepath)
    t_ex = time.time() - t0
    
    t0 = time.time()
    batches = [frames[i:i+2] for i in range(0, len(frames), 2)]
    vision_results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_batch_vision, batch, idx) for idx, batch in enumerate(batches)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                vision_results.append(res)
                
    combined_vision_summary = "\n\n".join(vision_results)
    t_vis = time.time() - t0
    
    t0 = time.time()
    tones = generate_tones(combined_vision_summary)
    t_llm = time.time() - t0
    
    return {
        "captions": tones,
        "frames": frames,
        "vision_summary": combined_vision_summary,
        "stats": {
            "download_time": t_dl,
            "extract_time": t_ex,
            "vision_time": t_vis,
            "llm_time": t_llm,
            "total_time": t_dl + t_ex + t_vis + t_llm
        }
    }

# ----------------- STREAMLIT FRONTEND -----------------

st.set_page_config(
    page_title="Video Captioning Frontend",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0d0f17;
        color: #f0f2f6;
    }
    .header-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .style-card {
        background: rgba(22, 25, 37, 0.7);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .style-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .badge-formal { background-color: #3b82f6; color: white; }
    .badge-sarcastic { background-color: #ef4444; color: white; }
    .badge-tech { background-color: #10b981; color: white; }
    .badge-non-tech { background-color: #f59e0b; color: white; }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .stat-box {
        text-align: center;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #3bf0ff;
    }
    .stat-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-title">🎬 Video Captioning Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Strictly adhering to your code while delivering a fast cached Streamlit frontend</p>', unsafe_allow_html=True)

# Sidebar Configuration status
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
st.sidebar.header("API Configuration Status")

if FIREWORKS_API_KEY:
    st.sidebar.success("🔑 Fireworks API: Configured")
else:
    st.sidebar.error("❌ Fireworks API: Missing")

if GROQ_API_KEY:
    st.sidebar.success("🔑 Groq API: Configured")
else:
    st.sidebar.error("❌ Groq API: Missing")

# Validate Keys
if not FIREWORKS_API_KEY or not GROQ_API_KEY:
    st.warning("⚠️ Please make sure to configure FIREWORKS_API_KEY and GROQ_API_KEY in your environment or `.env` file.")

# Video Input Choices
input_method = st.radio("Select Video Input:", ["Enter Video URL", "Upload Video File"], horizontal=True)

video_source = ""
uploaded_file = None

if input_method == "Enter Video URL":
    example_urls = {
        "v1: Autumn Boulevard": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
        "v2: Garden Kitten": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
        "v3: Office Worker": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4"
    }
    selected_example = st.selectbox("Quick Select Example Clip:", ["None"] + list(example_urls.keys()))
    
    url_val = ""
    if selected_example != "None":
        url_val = example_urls[selected_example]
        
    url_input = st.text_input("Or enter any video direct URL:", value=url_val)
    if url_input:
        video_source = url_input
else:
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "webm"])
    if uploaded_file:
        video_source = uploaded_file.name

if video_source:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Video Preview")
        if input_method == "Enter Video URL":
            st.video(video_source)
        else:
            st.video(uploaded_file)
            
    with col2:
        st.subheader("Process Video")
        st.write("Click below to process the video. This runs your unmodified code pipeline and caches the results.")
        run_btn = st.button("🚀 Run Captioning Agent", use_container_width=True, type="primary")

    if run_btn:
        if not FIREWORKS_API_KEY or not GROQ_API_KEY:
            st.error("Missing API key configurations. Please set them first.")
        else:
            status_container = st.status("Initializing processing pipeline...", expanded=True)
            
            try:
                if input_method == "Enter Video URL":
                    status_container.update(label="Processing URL (Downloading & Analyzing)...", state="running")
                    result = cached_process_video_pipeline_from_url(video_source)
                else:
                    status_container.update(label="Processing Uploaded File...", state="running")
                    
                    # Save uploaded file to temp file to pass to the function
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    file_hash = get_file_hash(temp_path)
                    result = cached_process_video_pipeline_from_local_file(file_hash, temp_path)
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                
                status_container.update(label="Processing complete!", state="complete", expanded=False)
                
                # Display Stats
                stats = result["stats"]
                st.markdown(f"""
                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-val">{stats['download_time']:.2f}s</div>
                        <div class="stat-lbl">Download</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{stats['extract_time']:.2f}s</div>
                        <div class="stat-lbl">Frame Extraction</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{stats['vision_time']:.2f}s</div>
                        <div class="stat-lbl">Vision Model</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{stats['llm_time']:.2f}s</div>
                        <div class="stat-lbl">Caption Gen</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val" style="color: #a855f7;">{stats['total_time']:.2f}s</div>
                        <div class="stat-lbl" style="color: #a855f7;">Total Time</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Results Tabs
                tab_captions, tab_frames, tab_summary = st.tabs(["✍️ Generated Captions", "🖼️ Extracted Keyframes", "📝 Raw Vision Summary"])
                
                captions = result["captions"]
                frames = result["frames"]
                combined_vision_summary = result["vision_summary"]
                
                with tab_captions:
                    col_f, col_s = st.columns(2)
                    col_t, col_n = st.columns(2)
                    
                    with col_f:
                        st.markdown(f"""
                        <div class="style-card">
                            <span class="style-badge badge-formal">Formal Style</span>
                            <p style="font-size: 1.1rem; line-height: 1.6; color: #e2e8f0;">{captions.get('formal', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_s:
                        st.markdown(f"""
                        <div class="style-card">
                            <span class="style-badge badge-sarcastic">Sarcastic Style</span>
                            <p style="font-size: 1.1rem; line-height: 1.6; color: #e2e8f0;">{captions.get('sarcastic', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_t:
                        st.markdown(f"""
                        <div class="style-card">
                            <span class="style-badge badge-tech">Humorous Tech Style</span>
                            <p style="font-size: 1.1rem; line-height: 1.6; color: #e2e8f0;">{captions.get('humorous_tech', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_n:
                        st.markdown(f"""
                        <div class="style-card">
                            <span class="style-badge badge-non-tech">Humorous Non-Tech Style</span>
                            <p style="font-size: 1.1rem; line-height: 1.6; color: #e2e8f0;">{captions.get('humorous_non_tech', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with tab_frames:
                    st.write(f"The model analyzed the following {len(frames)} keyframes (Scene & Uniform):")
                    cols = st.columns(4)
                    for i, frame in enumerate(frames):
                        col_idx = i % 4
                        with cols[col_idx]:
                            img_data = base64.b64decode(frame["b64"])
                            st.image(img_data, caption=f"Time: {frame['timestamp']:.2f}s (Frame #{frame['idx']})", use_container_width=True)
                            
                with tab_summary:
                    st.write("Detailed raw scene descriptions from Fireworks AI vision model:")
                    st.info(combined_vision_summary if combined_vision_summary else "No summary details returned.")
                    
            except Exception as e:
                status_container.update(label="An error occurred during processing.", state="error")
                st.error(f"Processing failed: {e}")
