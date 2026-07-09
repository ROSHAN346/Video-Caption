import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import time
import requests
from pathlib import Path

from scene_detector import detect_scenes
from frame_selector import select_keyframes
from config import (
    DETECTOR_CONFIG, FRAME_STRATEGY, MAX_FRAMES,
    CANDIDATE_FPS, CLIP_MODEL_NAME, EMBEDDING_BATCH_SIZE, EARLY_STOP_MIN_DIST,
    FIREWORKS_API_KEY, FIREWORKS_VISION_MODEL, FIREWORKS_TEXT_MODEL,
    GEMINI_API_KEY, GEMINI_VISION_MODEL, GEMINI_TEXT_MODEL,
    AI_PROVIDER,
    REPORT_STYLES, DEFAULT_REPORT_STYLE, REPORT_CACHE_ENABLED
)

st.set_page_config(
    page_title="Video Keyframe Extractor",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Video Keyframe Extractor")
st.markdown("Extract representative keyframes from videos using scene detection and CLIP embeddings.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Scene Detection")
adaptive_threshold = st.sidebar.slider(
    "Adaptive Threshold", 1.0, 10.0, DETECTOR_CONFIG["adaptive_threshold"], 0.1
)
min_scene_len = st.sidebar.slider(
    "Min Scene Length (frames)", 5, 100, DETECTOR_CONFIG["min_scene_len"]
)
window_width = st.sidebar.slider(
    "Window Width", 1, 10, DETECTOR_CONFIG["window_width"]
)
min_content_val = st.sidebar.slider(
    "Min Content Value", 5.0, 50.0, DETECTOR_CONFIG["min_content_val"], 0.5
)

st.sidebar.subheader("Keyframe Selection")
max_frames = st.sidebar.slider("Max Frames", 1, 50, MAX_FRAMES)
candidate_fps = st.sidebar.slider("Candidate FPS", 1.0, 10.0, CANDIDATE_FPS, 0.5)
early_stop_min_dist = st.sidebar.slider(
    "Early Stop Min Distance", 0.0, 0.5, EARLY_STOP_MIN_DIST, 0.01
)

st.sidebar.subheader("CLIP Model")
clip_model = st.sidebar.selectbox(
    "Model Variant",
    ["ViT-B/32", "ViT-B/16", "ViT-L/14"],
    index=0
)

st.sidebar.subheader("AI Analysis")
enable_reports = st.sidebar.checkbox("Enable AI Reports", value=False)

if enable_reports:
    provider_choice = st.sidebar.selectbox("AI Provider", ["gemini", "fireworks"], index=0 if AI_PROVIDER == "gemini" else 1)

    if provider_choice == "gemini":
        st.sidebar.info("Gemini AI via OpenAI-compatible endpoint.")
        api_token = st.sidebar.text_input(
            "Gemini API Key",
            value=GEMINI_API_KEY,
            type="password",
            help="Get yours at: https://aistudio.google.com/apikey"
        )
        vision_model = st.sidebar.selectbox(
            "Vision Model",
            [GEMINI_VISION_MODEL],
            index=0
        )
        text_model = st.sidebar.selectbox(
            "Text Model",
            [GEMINI_TEXT_MODEL],
            index=0
        )
    else:
        st.sidebar.info("Fireworks AI offers fast inference with competitive pricing.")
        api_token = st.sidebar.text_input(
            "Fireworks API Key",
            value=FIREWORKS_API_KEY,
            type="password",
            help="Get yours at: https://fireworks.ai/account/api-keys"
        )
        vision_model = st.sidebar.selectbox(
            "Vision Model",
            ["accounts/fireworks/models/minimax-m3", "accounts/fireworks/models/kimi-k2p7-code", "accounts/fireworks/models/llama-v3p2-11b-vision-instruct", "accounts/fireworks/models/llama-v3p2-90b-vision-instruct"],
            index=0
        )
        text_model = st.sidebar.selectbox(
            "Text Model",
            ["accounts/fireworks/models/qwen3-8b", "accounts/fireworks/models/llama-v3p1-8b-instruct", "accounts/fireworks/models/deepseek-v3p1", "accounts/fireworks/models/deepseek-v4-pro"],
            index=0
        )

    report_styles = st.sidebar.multiselect(
        "Report Styles",
        REPORT_STYLES,
        default=["formal", "sarcastic"]
    )

    generate_pdf = st.sidebar.checkbox("Generate PDFs", value=True)

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"]
    )

    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        # Display video info
        cap = cv2.VideoCapture(temp_video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            st.video(uploaded_file)

            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Resolution", f"{width}x{height}")
            with info_cols[1]:
                st.metric("FPS", f"{fps:.1f}")
            with info_cols[2]:
                st.metric("Duration", f"{duration:.1f}s")

        cap.release()

with col2:
    st.header("📊 Results")

    if uploaded_file is not None:
        run_pipeline = st.button("🚀 Extract Keyframes", type="primary", width='stretch')

        if run_pipeline:
            # Update config
            config = {
                "adaptive_threshold": adaptive_threshold,
                "min_scene_len": min_scene_len,
                "window_width": window_width,
                "min_content_val": min_content_val,
            }

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            timing_container = st.container()

            try:
                # Phase 1: Scene Detection
                status_text.text("🔍 Detecting scenes...")
                progress_bar.progress(10, text="Detecting scenes...")
                t_start = time.time()
                scenes = detect_scenes(temp_video_path)
                t_scenes = time.time() - t_start
                progress_bar.progress(30, text=f"Found {len(scenes)} scenes")
                with timing_container:
                    st.caption(f"⏱️ Scene detection: {t_scenes:.1f}s")

                # Phase 2: Keyframe Selection
                status_text.text("🎯 Selecting keyframes...")
                progress_bar.progress(40, text="Embedding frames with CLIP...")
                t_start = time.time()
                selected = select_keyframes(temp_video_path, scenes)
                t_keyframes = time.time() - t_start
                progress_bar.progress(80, text=f"Selected {len(selected)} keyframes")
                with timing_container:
                    st.caption(f"⏱️ Keyframe selection: {t_keyframes:.1f}s")

                # Phase 3: Save Results
                status_text.text("💾 Saving results...")
                t_save_start = time.time()
                output_dir = Path("output") / Path(uploaded_file.name).stem
                output_dir.mkdir(parents=True, exist_ok=True)

                MAX_SAVE_SIDE = 1280
                keyframes_data = []

                for i, sf in enumerate(selected):
                    filename = f"keyframe_{i:03d}.jpg"
                    filepath = output_dir / filename

                    # Downscale if needed
                    h, w = sf.image.shape[:2]
                    long_side = max(h, w)
                    if long_side > MAX_SAVE_SIDE:
                        scale = MAX_SAVE_SIDE / float(long_side)
                        image = cv2.resize(sf.image, (int(round(w * scale)), int(round(h * scale))),
                                          interpolation=cv2.INTER_AREA)
                    else:
                        image = sf.image

                    cv2.imwrite(str(filepath), image)
                    keyframes_data.append({
                        "frame_index": sf.frame_index,
                        "timestamp_sec": round(sf.timestamp_sec, 3),
                        "scene_id": sf.scene_id,
                        "novelty_score": round(sf.novelty_score, 4),
                        "image_path": filename,
                    })

                # Save JSON
                json_path = output_dir / "keyframes.json"
                with open(json_path, "w") as f:
                    json.dump(keyframes_data, f, indent=2)

                scenes_data = []
                for scene in scenes:
                    total_ms = int(scene.start_time * 1000)
                    hours = total_ms // 3600000
                    minutes = (total_ms % 3600000) // 60000
                    secs = (total_ms % 60000) // 1000
                    millis = total_ms % 1000
                    start_tc = f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

                    total_ms = int(scene.end_time * 1000)
                    hours = total_ms // 3600000
                    minutes = (total_ms % 3600000) // 60000
                    secs = (total_ms % 60000) // 1000
                    millis = total_ms % 1000
                    end_tc = f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

                    scenes_data.append({
                        "scene_number": scene.scene_number,
                        "start_time": start_tc,
                        "end_time": end_tc,
                        "duration": scene.duration,
                    })

                scenes_json_path = output_dir / "scenes.json"
                with open(scenes_json_path, "w") as f:
                    json.dump(scenes_data, f, indent=2)

                t_total = t_scenes + t_keyframes + (time.time() - t_save_start)
                progress_bar.progress(100, text="Complete!")
                status_text.success(f"✅ Extracted {len(selected)} keyframes from {len(scenes)} scenes")
                with timing_container:
                    st.caption(f"⏱️ Save results: {time.time() - t_save_start:.1f}s")
                    st.success(f"⏱️ **Total extraction time: {t_total:.1f}s**")

                # Store timing in session state
                st.session_state["t_scenes"] = t_scenes
                st.session_state["t_keyframes"] = t_keyframes

                # Store in session state for display
                st.session_state["selected"] = selected
                st.session_state["scenes"] = scenes
                st.session_state["keyframes_data"] = keyframes_data
                st.session_state["output_dir"] = output_dir

            except Exception as e:
                progress_bar.progress(100, text="Error occurred")
                status_text.error(f"❌ Error: {str(e)}")
                st.exception(e)

    # Report Generation Section
    if uploaded_file is not None and enable_reports and "selected" in st.session_state:
        st.divider()
        st.header("📝 AI Report Generation")

        if not api_token:
            st.warning(f"Please enter your {provider_choice.title()} API Key in the sidebar to enable report generation.")
        else:
            if st.button("🚀 Generate AI Reports", type="primary", width='stretch'):
                report_progress = st.progress(0, text="Initializing...")
                report_status = st.empty()
                report_timing = st.container()

                try:
                    if provider_choice == "gemini":
                        from services.gemini_client import GeminiClient
                        report_status.text("🔑 Initializing Gemini AI client...")
                        report_progress.progress(5, text="Connecting to Gemini AI...")
                        client = GeminiClient(api_token)
                    else:
                        from services.fireworks_client import FireworksClient
                        report_status.text("🔑 Initializing Fireworks AI client...")
                        report_progress.progress(5, text="Connecting to Fireworks AI...")
                        client = FireworksClient(api_token)
                    from services.image_analyzer import analyze_keyframe, save_analysis
                    from services.scene_aggregator import aggregate_by_scene
                    from services.report_generator import generate_all_reports
                    from services.report_cache import ReportCache
                    from services.pdf_generator import generate_report_pdf

                    selected = st.session_state["selected"]
                    scenes = st.session_state["scenes"]
                    output_dir = st.session_state["output_dir"]

                    cache = ReportCache(str(output_dir / "reports"))
                    analysis_dir = output_dir / "analysis"
                    analysis_dir.mkdir(parents=True, exist_ok=True)

                    # Analyze keyframes
                    report_status.text("🔍 Analyzing keyframes with vision model...")
                    analyses = []
                    t_vision_start = time.time()

                    for i, sf in enumerate(selected):
                        keyframe_path = str(output_dir / f"keyframe_{i:03d}.jpg")

                        if not Path(keyframe_path).exists():
                            continue

                        progress_pct = 10 + (i / len(selected)) * 40
                        report_progress.progress(int(progress_pct), text=f"Analyzing frame {i+1}/{len(selected)}...")

                        try:
                            analysis = analyze_keyframe(
                                image_path=keyframe_path,
                                hf_client=client,
                                model=vision_model,
                                scene_id=sf.scene_id,
                                frame_index=i
                            )
                            # Skip failed analyses
                            if analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
                                continue
                            analyses.append(analysis)
                        except Exception as e:
                            continue

                        # Save individual analysis
                        analysis_path = analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
                        save_analysis(analysis, str(analysis_path))

                    t_vision_done = time.time() - t_vision_start
                    with report_timing:
                        st.caption(f"⏱️ Vision analysis: {t_vision_done:.1f}s ({len(analyses)}/{len(selected)} frames)")

                    if not analyses:
                        report_status.warning("Vision analysis unavailable. Using scene data as fallback.")
                        # Create fallback analysis from scene detection
                        for sf in selected[:1]:
                            fallback = {
                                "scene_id": sf.scene_id,
                                "frame_index": 0,
                                "scene_type": "unknown",
                                "location": "unknown",
                                "people": "unknown",
                                "objects": [],
                                "vehicles": [],
                                "animals": [],
                                "activities": "video content",
                                "weather": "unknown",
                                "time_of_day": "unknown",
                                "environment": "unknown",
                                "risk_level": "low",
                                "confidence": 0.5,
                                "summary": f"Scene {sf.scene_id} spanning {scenes[0].duration:.1f} seconds with {len(selected)} keyframes extracted."
                            }
                            analyses.append(fallback)

                    if analyses:
                        # Aggregate by scene
                        report_status.text("📊 Aggregating analyses by scene...")
                        report_progress.progress(55, text="Aggregating...")
                        scene_analyses = aggregate_by_scene(analyses, scenes)

                        # Generate reports
                        report_status.text("✍️ Generating reports...")
                        total_scenes = len(scene_analyses)
                        t_report_start = time.time()
                        for idx, (scene_id, scene_data) in enumerate(scene_analyses.items()):
                            progress_pct = 60 + (idx / total_scenes) * 35
                            report_progress.progress(int(progress_pct), text=f"Generating reports for scene {scene_id}...")

                            reports = generate_all_reports(scene_data, client, report_styles, text_model)

                            for style, report in reports.items():
                                cache.cache_report(scene_id, style, report, {
                                    "vision_model": vision_model,
                                    "text_model": text_model,
                                    "provider": provider_choice
                                })

                                if generate_pdf:
                                    keyframe_path = str(output_dir / f"keyframe_000.jpg")
                                    pdf_path = cache.get_pdf_path(scene_id, style)
                                    generate_report_pdf(
                                        scene_id=scene_id,
                                        style=style,
                                        report=report,
                                        keyframe_path=keyframe_path,
                                        scene_data=scene_data,
                                        output_path=str(pdf_path)
                                    )

                        t_report_done = time.time() - t_report_start
                        t_total_report = t_vision_done + t_report_done

                        report_progress.progress(100, text="Complete!")
                        report_status.success(f"✅ Generated {len(reports)} report styles for {total_scenes} scenes")
                        with report_timing:
                            st.caption(f"⏱️ Report generation: {t_report_done:.1f}s")
                            st.success(f"⏱️ **Total report time: {t_total_report:.1f}s** (Vision: {t_vision_done:.1f}s + Text: {t_report_done:.1f}s)")

                        # Store timing and reports in session state
                        st.session_state["t_vision"] = t_vision_done
                        st.session_state["t_report"] = t_report_done
                        st.session_state["reports"] = reports
                        st.session_state["scene_analyses"] = scene_analyses

                except Exception as e:
                    report_progress.progress(100, text="Error occurred")
                    report_status.error(f"❌ Error: {str(e)}")
                    st.exception(e)

    else:
        st.info("👆 Upload a video file to get started")

# --- Batch Processing Section ---
st.divider()
st.header("📋 Batch Processing (JSON Upload)")
st.markdown("Upload a JSON file with multiple video URLs to process them all at once.")

batch_json_file = st.file_uploader(
    "Upload tasks.json file",
    type=["json"],
    key="batch_json"
)

if batch_json_file is not None:
    try:
        tasks = json.load(batch_json_file)
        st.success(f"Loaded {len(tasks)} tasks from JSON file")

        # Display tasks in a table
        task_data = []
        for task in tasks:
            task_data.append({
                "Task ID": task.get("task_id", "N/A"),
                "Video URL": task.get("video_url", "N/A")[:60] + "..." if len(task.get("video_url", "")) > 60 else task.get("video_url", "N/A"),
                "Styles": ", ".join(task.get("styles", []))
            })
        st.dataframe(task_data, width='stretch')

        # API key input for batch processing
        batch_api_key = st.text_input(
            f"{provider_choice.title()} API Key",
            value=api_token if enable_reports else "",
            type="password",
            key="batch_api_key",
            help="Required for AI analysis of videos"
        )

        # Process batch button
        if st.button("🚀 Process All Videos", type="primary", key="process_batch", width='stretch'):
            if not batch_api_key:
                st.error(f"Please enter your {provider_choice.title()} API Key to process videos.")
            else:
                # Initialize client
                if provider_choice == "gemini":
                    from services.gemini_client import GeminiClient
                    client = GeminiClient(batch_api_key)
                else:
                    from services.fireworks_client import FireworksClient
                    client = FireworksClient(batch_api_key)

                batch_progress = st.progress(0, text="Starting batch processing...")
                batch_status = st.empty()
                batch_results = []

                for task_idx, task in enumerate(tasks):
                    task_id = task.get("task_id", f"task_{task_idx}")
                    video_url = task.get("video_url", "")
                    styles = task.get("styles", REPORT_STYLES)

                    batch_status.text(f"📥 Processing task {task_id} ({task_idx + 1}/{len(tasks)})...")

                    try:
                        # Download video
                        batch_progress.progress(int((task_idx / len(tasks)) * 100), text=f"Downloading video {task_id}...")
                        response = requests.get(video_url, timeout=120, stream=True)
                        response.raise_for_status()

                        # Determine file extension
                        suffix = ".mp4"
                        url_path = video_url.split("?")[0] if "?" in video_url else video_url
                        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                            if url_path.lower().endswith(ext):
                                suffix = ext
                                break

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                tmp_file.write(chunk)
                            temp_video_path = tmp_file.name

                        # Run pipeline
                        batch_status.text(f"🔍 Detecting scenes for {task_id}...")
                        scenes = detect_scenes(temp_video_path)

                        batch_status.text(f"🎯 Selecting keyframes for {task_id}...")
                        selected = select_keyframes(temp_video_path, scenes)

                        # Save keyframes
                        output_dir = Path(tempfile.mkdtemp(prefix=f"batch_{task_id}_"))
                        output_dir.mkdir(parents=True, exist_ok=True)
                        analysis_dir = output_dir / "analysis"
                        analysis_dir.mkdir(parents=True, exist_ok=True)

                        MAX_SAVE_SIDE = 1280
                        for i, sf in enumerate(selected):
                            filepath = output_dir / f"keyframe_{i:03d}.jpg"
                            h, w = sf.image.shape[:2]
                            long_side = max(h, w)
                            if long_side > MAX_SAVE_SIDE:
                                scale = MAX_SAVE_SIDE / float(long_side)
                                image = cv2.resize(sf.image, (int(round(w * scale)), int(round(h * scale))),
                                                  interpolation=cv2.INTER_AREA)
                            else:
                                image = sf.image
                            cv2.imwrite(str(filepath), image)

                        # Analyze keyframes with retry
                        batch_status.text(f"🔍 Analyzing keyframes for {task_id}...")
                        analyses = []

                        for i, sf in enumerate(selected):
                            keyframe_path = str(output_dir / f"keyframe_{i:03d}.jpg")
                            if not Path(keyframe_path).exists():
                                continue

                            for attempt in range(3):
                                try:
                                    analysis = analyze_keyframe(
                                        image_path=keyframe_path,
                                        hf_client=client,
                                        model=vision_model,
                                        scene_id=sf.scene_id,
                                        frame_index=i
                                    )
                                    if analysis.get("summary") not in ["Analysis unavailable", "Analysis failed to parse", ""]:
                                        analyses.append(analysis)
                                        save_analysis(analysis, str(analysis_dir / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"))
                                    break
                                except Exception as e:
                                    if "429" in str(e) or "rate" in str(e).lower():
                                        time.sleep(5 * (2 ** attempt))
                                    else:
                                        break

                        # Fallback if no analyses
                        if not analyses:
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

                        # Aggregate and generate captions
                        scene_analyses = aggregate_by_scene(analyses, scenes)

                        batch_status.text(f"✍️ Generating captions for {task_id}...")
                        captions = {}
                        for style in styles:
                            try:
                                report = generate_video_summary_reports(scene_analyses, client, [style], text_model)
                                captions[style] = report.get(style, f"Failed to generate {style} caption")
                            except Exception as e:
                                captions[style] = f"Error generating {style} caption"

                        # Cleanup temp files
                        try:
                            os.unlink(temp_video_path)
                        except OSError:
                            pass

                        batch_results.append({
                            "task_id": task_id,
                            "captions": captions,
                            "scenes": len(scenes),
                            "keyframes": len(selected),
                            "analyses": len(analyses)
                        })

                        batch_status.success(f"✅ Completed task {task_id}")

                    except Exception as e:
                        batch_results.append({
                            "task_id": task_id,
                            "captions": {style: f"Failed to process video" for style in styles},
                            "error": str(e)
                        })
                        batch_status.error(f"❌ Failed task {task_id}: {str(e)}")

                # Display batch results
                batch_progress.progress(100, text="Batch processing complete!")
                st.success(f"✅ Processed {len(batch_results)} tasks")

                # Store results in session state
                st.session_state["batch_results"] = batch_results

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")

# --- Display Batch Results ---
if "batch_results" in st.session_state:
    st.divider()
    st.header("📊 Batch Processing Results")

    batch_results = st.session_state["batch_results"]

    # Display results for each task
    for result in batch_results:
        task_id = result.get("task_id", "Unknown")
        with st.expander(f"Task: {task_id}", expanded=True):
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scenes", result.get("scenes", 0))
                with col2:
                    st.metric("Keyframes", result.get("keyframes", 0))
                with col3:
                    st.metric("Analyses", result.get("analyses", 0))

                st.subheader("Captions")
                captions = result.get("captions", {})
                for style, caption in captions.items():
                    st.markdown(f"**{style.replace('_', ' ').title()}:** {caption}")

    # Download batch results as JSON
    st.divider()
    st.header("📥 Download Results")

    # Format for competition output
    competition_output = []
    for result in batch_results:
        competition_output.append({
            "task_id": result.get("task_id"),
            "captions": result.get("captions", {})
        })

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download results.json (Competition Format)",
            data=json.dumps(competition_output, indent=2),
            file_name="results.json",
            mime="application/json"
        )
    with col2:
        st.download_button(
            label="📄 Download full results (with metadata)",
            data=json.dumps(batch_results, indent=2),
            file_name="full_results.json",
            mime="application/json"
        )

# --- Display Results ---
if "selected" in st.session_state:
    st.divider()
    st.header("🖼️ Extracted Keyframes")

    selected = st.session_state["selected"]
    keyframes_data = st.session_state["keyframes_data"]
    output_dir = st.session_state["output_dir"]

    # Display timing metrics
    if "t_scenes" in st.session_state:
        st.subheader("⏱️ Processing Time")
        timing_cols = st.columns(4)
        with timing_cols[0]:
            st.metric("Scene Detection", f"{st.session_state['t_scenes']:.1f}s")
        with timing_cols[1]:
            st.metric("Keyframe Selection", f"{st.session_state['t_keyframes']:.1f}s")
        if "t_vision" in st.session_state:
            with timing_cols[2]:
                st.metric("Vision Analysis", f"{st.session_state['t_vision']:.1f}s")
        if "t_report" in st.session_state:
            with timing_cols[3]:
                st.metric("Report Generation", f"{st.session_state['t_report']:.1f}s")

    # Display keyframes in grid
    cols_per_row = 4
    for row_start in range(0, len(selected), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            if row_start + idx < len(selected):
                sf = selected[row_start + idx]
                data = keyframes_data[row_start + idx]

                with col:
                    # Convert BGR to RGB for display
                    rgb_image = cv2.cvtColor(sf.image, cv2.COLOR_BGR2RGB)
                    st.image(
                        rgb_image,
                        caption=f"Frame {data['frame_index']} | Scene {data['scene_id']}\n"
                                f"Time: {data['timestamp_sec']:.2f}s | Novelty: {data['novelty_score']:.3f}",
                        width='stretch'
                    )

    # Scene breakdown
    st.divider()
    st.header("📋 Scene Breakdown")

    scenes = st.session_state["scenes"]
    scene_df_data = []
    for scene in scenes:
        scene_df_data.append({
            "Scene": scene.scene_number,
            "Start": f"{scene.start_time:.2f}s",
            "End": f"{scene.end_time:.2f}s",
            "Duration": f"{scene.duration:.2f}s",
            "Frames": f"{scene.start_frame}-{scene.end_frame}"
        })

    st.dataframe(scene_df_data, width='stretch')

    # Download results
    st.divider()
    st.header("📥 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        with open(output_dir / "keyframes.json", "r") as f:
            st.download_button(
                label="📄 Download keyframes.json",
                data=f.read(),
                file_name="keyframes.json",
                mime="application/json"
            )

    with col2:
        with open(output_dir / "scenes.json", "r") as f:
            st.download_button(
                label="📄 Download scenes.json",
                data=f.read(),
                file_name="scenes.json",
                mime="application/json"
            )

# --- Report Display Section ---
if "reports" in st.session_state and "scene_analyses" in st.session_state:
    st.divider()
    st.header("📝 Generated Reports")

    # Show timing summary if available
    if "t_vision" in st.session_state and "t_report" in st.session_state:
        total_time = st.session_state.get("t_scenes", 0) + st.session_state.get("t_keyframes", 0) + st.session_state["t_vision"] + st.session_state["t_report"]
        st.info(f"⏱️ Total pipeline time: **{total_time:.1f}s** (Scenes: {st.session_state.get('t_scenes', 0):.1f}s | Keyframes: {st.session_state.get('t_keyframes', 0):.1f}s | Vision: {st.session_state['t_vision']:.1f}s | Reports: {st.session_state['t_report']:.1f}s)")

    reports = st.session_state["reports"]
    scene_analyses = st.session_state["scene_analyses"]

    # Tabs for each style
    tab_names = [s.replace("_", " ").title() for s in reports.keys()]
    tabs = st.tabs(tab_names)

    for tab, (style, report) in zip(tabs, reports.items()):
        with tab:
            st.markdown(report)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    f"📄 Download {style}.md",
                    report,
                    file_name=f"{style}.md",
                    mime="text/markdown",
                    key=f"dl_md_{style}"
                )
            with col2:
                if "generate_pdf" in dir() and generate_pdf:
                    pdf_path = output_dir / "reports" / "scene_001" / f"{style}.pdf"
                    if pdf_path.exists():
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                f"📥 Download {style}.pdf",
                                f.read(),
                                file_name=f"{style}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{style}"
                            )

    # Scene Analysis Details
    st.divider()
    st.header("🔍 Scene Analysis Details")

    for scene_id, scene_data in scene_analyses.items():
        with st.expander(f"Scene {scene_id}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Scene Type:**", scene_data.get("scene_type", "N/A"))
                st.write("**Location:**", scene_data.get("location", "N/A"))
                st.write("**People:**", scene_data.get("people", "N/A"))
                st.write("**Weather:**", scene_data.get("weather", "N/A"))

            with col2:
                st.write("**Time of Day:**", scene_data.get("time_of_day", "N/A"))
                st.write("**Environment:**", scene_data.get("environment", "N/A"))
                st.write("**Risk Level:**", scene_data.get("risk_level", "N/A"))
                st.write("**Confidence:**", f"{scene_data.get('confidence', 0):.2f}")

            st.write("**Objects:**", ", ".join(scene_data.get("objects", [])))
            st.write("**Activities:**", scene_data.get("activities", "N/A"))
            st.write("**Summary:**", scene_data.get("summary", "N/A"))

    # Cleanup temp file
    if "temp_video_path" in locals():
        try:
            os.unlink(temp_video_path)
        except:
            pass
