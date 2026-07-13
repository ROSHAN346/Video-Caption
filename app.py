"""Streamlit UI — Material Design 3 with Ant Design Components.

Usage:
    pip install streamlit streamlit-antd-components
    streamlit run app.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit_antd_components as sac

try:
    import config as cfg
except Exception:
    cfg = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _video_short_name(video_path: str) -> str:
    stem = Path(video_path).stem
    prefix = stem.split("-", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", prefix).strip("._-")
    return safe or stem


# ---------------------------------------------------------------------------
# Page Config & Material Design 3 Theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="No-Cap AI | Video Captioning", layout="wide", page_icon="🎬")

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* ===== Material Design 3 — Dark Theme ===== */
        * { font-family: 'Roboto', sans-serif; }

        :root {
            --md-bg: #121212;
            --md-surface-1: #1e1e1e;
            --md-surface-2: #242424;
            --md-surface-3: #2c2c2c;
            --md-surface-4: #353535;
            --md-primary: #6750a4;
            --md-primary-light: #d0bcff;
            --md-on-surface: #e6e1e5;
            --md-on-surface-variant: #c4c6d0;
            --md-outline: #49454f;
            --md-error: #cf6679;
            --md-success: #81c784;
            --md-warning: #ffb74d;
        }

        /* ===== Layout ===== */
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }

        /* ===== App Bar / Header ===== */
        .md-app-bar {
            background: var(--md-surface-1);
            border-radius: 0 0 24px 24px;
            padding: 1.75rem 2rem;
            margin: -2.5rem -2rem 2rem -2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border-bottom: 1px solid var(--md-outline);
        }
        .md-app-title {
            font-size: 2rem;
            font-weight: 700;
            color: var(--md-primary-light);
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .md-app-subtitle {
            font-size: 0.95rem;
            font-weight: 400;
            color: var(--md-on-surface-variant);
            margin-top: 0.4rem;
        }

        /* ===== Material Cards (Elevation) ===== */
        .md-card {
            background: var(--md-surface-2);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .md-card-elevated {
            background: var(--md-surface-3);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .md-card-outlined {
            background: var(--md-surface-1);
            border: 1px solid var(--md-outline);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
        }

        /* ===== Typography ===== */
        .md-headline {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--md-on-surface);
            margin: 0 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .md-title {
            font-size: 1.15rem;
            font-weight: 500;
            color: var(--md-on-surface);
            margin: 0 0 0.75rem 0;
        }
        .md-body {
            font-size: 0.95rem;
            font-weight: 400;
            color: var(--md-on-surface-variant);
            line-height: 1.6;
        }
        .md-label {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--md-on-surface-variant);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        /* ===== Caption Display ===== */
        .md-caption-box {
            background: var(--md-surface-2);
            border-left: 4px solid var(--md-primary);
            border-radius: 4px 12px 12px 4px;
            padding: 1.25rem 1.5rem;
            color: var(--md-on-surface);
            line-height: 1.75;
            font-size: 0.98rem;
        }

        /* ===== Keyframe Grid ===== */
        .md-keyframe {
            background: var(--md-surface-2);
            border-radius: 12px;
            padding: 0.65rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .md-keyframe:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            transform: translateY(-3px);
        }

        /* ===== Stat Cards ===== */
        .md-stat {
            background: var(--md-surface-2);
            border-radius: 16px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .md-stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--md-primary-light);
            font-family: 'Roboto Mono', monospace;
        }
        .md-stat-label {
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--md-on-surface-variant);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }

        /* ===== Status Chips ===== */
        .md-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.85rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 500;
        }
        .md-chip-success { background: rgba(129,199,132,0.15); color: var(--md-success); border: 1px solid rgba(129,199,132,0.3); }
        .md-chip-error   { background: rgba(207,102,121,0.15); color: var(--md-error); border: 1px solid rgba(207,102,121,0.3); }
        .md-chip-info    { background: rgba(208,188,255,0.15); color: var(--md-primary-light); border: 1px solid rgba(208,188,255,0.3); }
        .md-chip-warning { background: rgba(255,183,77,0.15); color: var(--md-warning); border: 1px solid rgba(255,183,77,0.3); }

        /* ===== Section Divider ===== */
        .md-divider {
            height: 1px;
            background: var(--md-outline);
            margin: 2.5rem 0;
            border: none;
        }
        .md-gap { margin-top: 2rem; }

        /* ===== Streamlit Overrides ===== */
        /* Buttons — Material filled tonal */
        .stButton > button {
            background: var(--md-primary) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 0.6rem 1.75rem !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 2px 8px rgba(103,80,164,0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #7c5cb0 !important;
            box-shadow: 0 4px 12px rgba(103,80,164,0.4) !important;
        }

        /* File uploader */
        .stFileUploader > div > div {
            background: var(--md-surface-2) !important;
            border: 2px dashed var(--md-outline) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
        }
        .stFileUploader > div > div:hover {
            border-color: var(--md-primary) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--md-surface-1);
            border-radius: 12px;
            padding: 0.35rem;
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 20px;
            color: var(--md-on-surface-variant);
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.45rem 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--md-primary);
            color: #fff;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--md-surface-1);
            border-right: 1px solid var(--md-outline);
        }

        /* Expander */
        details {
            background: var(--md-surface-1);
            border: 1px solid var(--md-outline) !important;
            border-radius: 12px !important;
        }
        summary {
            font-weight: 500 !important;
            color: var(--md-on-surface) !important;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background: var(--md-surface-2);
            border: 1px solid var(--md-outline);
            border-radius: 12px;
        }

        /* Code blocks */
        .stCodeBlock {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Hide branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# App Bar Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="md-app-bar">
        <div class="md-app-title">🎬 No-Cap AI</div>
        <div class="md-app-subtitle">AI-powered video captioning pipeline — scene detection, keyframe extraction & multi-style generation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Configuration")

    api_key_input = st.text_input(
        "Fireworks API Key (Optional)",
        value=os.getenv("FIREWORKS_API_KEY", ""),
        type="password",
        help="Direct Fireworks API key. Optional if using default backend.",
    )

    worker_url_input = os.getenv("WORKER_URL", "https://patient-violet-5828.viratforedu175.workers.dev")

    if worker_url_input:
        os.environ["WORKER_URL"] = worker_url_input
    if api_key_input:
        os.environ["FIREWORKS_API_KEY"] = api_key_input
        os.environ["FIREWORKS_TEXT_API_KEY"] = api_key_input

    st.divider()

    if cfg is not None:
        with st.expander("Model & Parameters", expanded=False):
            st.markdown("**Active Models**")
            model_rows = [
                ("Vision", getattr(cfg, "FIREWORKS_VISION_MODEL", "")),
                ("Text", getattr(cfg, "FIREWORKS_TEXT_MODEL", "")),
            ]
            for name, val in model_rows:
                c1, c2 = st.columns([1, 1.8])
                c1.markdown(f"<span class='md-label'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)

            st.markdown("**Parameters**")
            param_rows = [
                ("Max frames", getattr(cfg, "MAX_FRAMES", "")),
                ("Candidate FPS", getattr(cfg, "CANDIDATE_FPS", "")),
                ("Batch size", getattr(cfg, "EMBEDDING_BATCH_SIZE", "")),
            ]
            for name, val in param_rows:
                c1, c2 = st.columns([1, 1.8])
                c1.markdown(f"<span class='md-label'>{name}</span>", unsafe_allow_html=True)
                c2.code(str(val), language=None)


# ---------------------------------------------------------------------------
# Results (Batch mode)
# ---------------------------------------------------------------------------
def show_results(results: list, times: dict):
    st.markdown("<hr class='md-divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='md-headline'>Results</h2>", unsafe_allow_html=True)

    if not results:
        st.warning("No results produced.")
        return

    task_times = {k: v for k, v in times.items() if not k.startswith("__")}
    done_times = [v["time"] for v in task_times.values() if v.get("state") == "done"]
    failed = [tid for tid, v in task_times.items() if v.get("state") == "failed"]
    overall = times.get("__total__", {}).get("time")
    total_time = overall if overall is not None else sum(done_times)
    avg_time = (total_time / len(task_times)) if task_times else 0.0

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='md-stat'><div class='md-stat-value'>{len(results)}</div><div class='md-stat-label'>Tasks</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='md-stat'><div class='md-stat-value'>{total_time:.1f}s</div><div class='md-stat-label'>Total Time</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='md-stat'><div class='md-stat-value'>{avg_time:.1f}s</div><div class='md-stat-label'>Avg / Video</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='md-stat'><div class='md-stat-value'>{len(failed)}</div><div class='md-stat-label'>Failed</div></div>", unsafe_allow_html=True)

    # Per-video results
    st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='md-title'>Per-video captions</h3>", unsafe_allow_html=True)
    task_ids = [r.get("task_id", f"task_{i}") for i, r in enumerate(results)]
    selected = st.selectbox("Select a video / task", task_ids, index=0)

    idx = task_ids.index(selected)
    result = results[idx]

    st.markdown("<div class='md-card-elevated'>", unsafe_allow_html=True)
    st.markdown(f"### Task: `{selected}`")

    if selected in times:
        t = times[selected]
        if t.get("state") == "done":
            sac.badge(label=f"Completed in {t['time']:.1f}s", color='green', variant='fill')
        elif t.get("state") == "failed":
            sac.badge(label="Failed", color='red', variant='fill')

    captions = result.get("captions", {})
    if not captions:
        st.info("No captions were generated for this task.")
    else:
        style_names = list(captions.keys())
        tabs = st.tabs(style_names)
        for tab, style in zip(tabs, style_names):
            with tab:
                text = captions[style]
                if isinstance(text, str):
                    st.markdown(f"<div class='md-caption-box'>{text}</div>", unsafe_allow_html=True)
                else:
                    st.json(text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
    st.download_button(
        "Download results.json",
        data=json.dumps(results, indent=2),
        file_name="results.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single Video", "Batch JSON"])

# ===========================================================================
# TAB 1: Single Video
# ===========================================================================
with tab_single:
    st.markdown(
        "<div class='md-card-outlined'>"
        "<h3 class='md-title'>Process a single video file</h3>"
        "<p class='md-body'>Upload a video. The pipeline splits it into scenes, "
        "selects keyframes, and queries the vision-language model for multi-style captions.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Max file size ~500MB",
    )

    if uploaded_video is not None:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("<div class='md-card'>", unsafe_allow_html=True)
            st.video(uploaded_video)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='md-card'>", unsafe_allow_html=True)
            st.markdown("#### Execution Parameters")
            style_choices = ["All Styles", "formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
            selected_style = st.selectbox(
                "Caption Tone",
                style_choices,
                index=0,
                help="'All Styles' generates captions in 4 different tones simultaneously!",
            )
            run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_video.read())
                temp_video_path = tmp.name

            short_name = _video_short_name(temp_video_path)

            cmd = [sys.executable, "main.py", temp_video_path, "--reports"]
            if selected_style == "All Styles":
                cmd.append("--all-styles")
            else:
                cmd.extend(["--style", selected_style])

            log_box = st.expander("Console Logs", expanded=True)
            log_container = log_box.empty()
            log_lines = []

            status_placeholder = st.empty()
            status_placeholder.info("Initializing pipeline...")

            # Pipeline steps indicator
            steps_placeholder = st.empty()

            try:
                sub_env = os.environ.copy()
                if worker_url_input:
                    sub_env["WORKER_URL"] = worker_url_input
                if api_key_input:
                    sub_env["FIREWORKS_API_KEY"] = api_key_input
                    sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=sub_env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                current_step = 0
                for raw in proc.stdout:
                    line = raw.strip()
                    if line:
                        log_lines.append(line)
                        log_container.code("\n".join(log_lines[-25:]), language="text")

                        if "Scene Detection" in line or "detect_scenes" in line:
                            current_step = 1
                            status_placeholder.info("Detecting scenes & capturing frame candidates...")
                        elif "select_keyframes" in line or "Selected keyframes" in line:
                            current_step = 2
                            status_placeholder.info("Selecting keyframes...")
                        elif "Analyzing" in line and "vision" in line:
                            current_step = 3
                            status_placeholder.info("Analyzing keyframes with Fireworks vision model...")
                        elif "Generating" in line and ("styles" in line or "report" in line.lower()):
                            current_step = 4
                            status_placeholder.info("Generating multi-style captions...")

                        # Update steps indicator
                        with steps_placeholder.container():
                            sac.steps(
                                steps=[
                                    sac.Steps(title="Scene Detection", description="Split video into scenes", status="wait" if current_step < 1 else ("finish" if current_step > 1 else "process")),
                                    sac.Steps(title="Keyframe Selection", description="Extract representative frames", status="wait" if current_step < 2 else ("finish" if current_step > 2 else "process")),
                                    sac.Steps(title="Vision Analysis", description="Analyze frames with AI", status="wait" if current_step < 3 else ("finish" if current_step > 3 else "process")),
                                    sac.Steps(title="Caption Generation", description="Generate styled captions", status="wait" if current_step < 4 else ("finish" if current_step > 4 else "process")),
                                ],
                                size='small',
                                direction='horizontal',
                                current=current_step,
                            )

                proc.wait()

                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

                if proc.returncode == 0:
                    status_placeholder.success("Pipeline completed successfully!")
                    with steps_placeholder.container():
                        sac.steps(
                            steps=[
                                sac.Steps(title="Scene Detection", description="Split video into scenes", status="finish"),
                                sac.Steps(title="Keyframe Selection", description="Extract representative frames", status="finish"),
                                sac.Steps(title="Vision Analysis", description="Analyze frames with AI", status="finish"),
                                sac.Steps(title="Caption Generation", description="Generate styled captions", status="finish"),
                            ],
                            size='small',
                            direction='horizontal',
                            current=4,
                        )

                    output_dir_path = Path("output") / short_name
                    captions_file = output_dir_path / "captions.json"

                    if captions_file.exists():
                        with open(captions_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            st.session_state["single_result"] = data
                            st.session_state["single_short_name"] = short_name
                    else:
                        st.error("Output captions.json was not created by the pipeline.")
                else:
                    status_placeholder.error(f"Pipeline crashed with return code {proc.returncode}.")
            except Exception as e:
                status_placeholder.error(f"Error during execution: {e}")

        # Results display
        if "single_result" in st.session_state:
            data = st.session_state["single_result"]
            short_name = st.session_state["single_short_name"]

            st.markdown("<hr class='md-divider'>", unsafe_allow_html=True)
            st.markdown("<h2 class='md-headline'>Video Analysis Results</h2>", unsafe_allow_html=True)

            # Captions
            st.markdown("<h3 class='md-title'>Generated Captions</h3>", unsafe_allow_html=True)
            captions = data.get("captions", {})
            if not captions:
                st.warning("No captions generated.")
            else:
                style_names = list(captions.keys())
                tabs = st.tabs(style_names)
                for tab, (style, text) in zip(tabs, captions.items()):
                    with tab:
                        st.markdown(f"<div class='md-caption-box'>{text}</div>", unsafe_allow_html=True)

            # Keyframes
            st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
            st.markdown("<h3 class='md-title'>Keyframe Selection</h3>", unsafe_allow_html=True)
            st.markdown("<p class='md-body'>Representative frames selected from the video for analysis.</p>", unsafe_allow_html=True)

            keyframes = data.get("keyframes", [])
            if keyframes:
                cols = st.columns(min(len(keyframes), 5))
                for i, kf in enumerate(keyframes):
                    img_path = Path("output") / short_name / kf.get("image_path", "")
                    with cols[i % len(cols)]:
                        st.markdown(f"<div class='md-keyframe'>", unsafe_allow_html=True)
                        if img_path.exists():
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.info("Image not found")
                        st.markdown(
                            f"<span class='md-label'>Scene {kf.get('scene_id')}</span><br>"
                            f"<span style='color:var(--md-primary-light);font-weight:500;'>"
                            f"{kf.get('timestamp_sec')}s</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

            # Scenes
            st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
            with st.expander("Detected Scenes Timeline"):
                scenes = data.get("scenes", [])
                if scenes:
                    st.table(scenes)
                else:
                    st.write("No scenes detected.")

            # Download
            st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
            st.download_button(
                "Download captions.json",
                data=json.dumps(data, indent=2),
                file_name="captions.json",
                mime="application/json",
            )


# ===========================================================================
# TAB 2: Batch JSON
# ===========================================================================
with tab_batch:
    st.markdown(
        "<div class='md-card-outlined'>"
        "<h3 class='md-title'>Run pipeline on a batch tasks JSON</h3>"
        "<p class='md-body'>Provide a <code>tasks.json</code> containing URLs of videos, styles, and task IDs. "
        "The pipeline processes them in parallel and downloads the final <code>results.json</code>.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_batch = st.file_uploader("Upload tasks JSON", type=["json"], key="batch_uploader")

    if uploaded_batch is not None:
        try:
            tasks = json.loads(uploaded_batch.getvalue().decode("utf-8"))
            if not isinstance(tasks, list):
                st.error("Top-level JSON must be a list of tasks.")
                st.stop()
            sac.alert(
                title=f"Loaded {len(tasks)} task(s)",
                description="Tasks JSON parsed successfully",
                status='success',
                variant='fill',
            )
            with st.expander("Preview uploaded JSON"):
                st.json(tasks)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        st.markdown("<div class='md-gap'></div>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        if run_col.button("Run Batch Pipeline", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                input_path = os.path.join(tmp, "tasks.json")
                with open(input_path, "w", encoding="utf-8") as f:
                    f.write(uploaded_batch.getvalue().decode("utf-8"))

                output_path = os.path.join(tmp, "results.json")
                cmd = [sys.executable, "main.py", "--input", input_path, "--output", output_path]

                status = {}
                status_box = st.empty()
                log_box = st.expander("Full process log", expanded=False)
                log_text = []

                def render_status():
                    if not status:
                        status_box.info("Waiting for tasks to start...")
                        return
                    items = []
                    for tid, info in status.items():
                        if info["state"] == "running":
                            items.append(sac.Steps(title=tid, description="Processing...", status="process"))
                        elif info["state"] == "done":
                            items.append(sac.Steps(title=tid, description=f"{info['time']:.1f}s", status="finish"))
                        else:
                            items.append(sac.Steps(title=tid, description="Failed", status="error"))
                    with status_box.container():
                        sac.steps(steps=items, size='small', direction='vertical', current=-1)

                render_status()

                try:
                    sub_env = os.environ.copy()
                    if worker_url_input:
                        sub_env["WORKER_URL"] = worker_url_input
                    if api_key_input:
                        sub_env["FIREWORKS_API_KEY"] = api_key_input
                        sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=sub_env,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                    )

                    re_start = re.compile(r"=== Processing task (.+?) ===")
                    re_done = re.compile(r"Task (.+?) completed in ([\d.]+)s")
                    re_fail = re.compile(r"Task (.+?) failed")
                    re_all = re.compile(r"All tasks completed in ([\d.]+)s")

                    for raw in proc.stdout:
                        line = raw.strip()
                        if not line:
                            continue
                        log_text.append(line)

                        m = re_start.search(line)
                        if m:
                            status[m.group(1)] = {"state": "running"}
                            render_status()
                            continue
                        m = re_done.search(line)
                        if m:
                            status[m.group(1)] = {"state": "done", "time": float(m.group(2))}
                            render_status()
                            continue
                        m = re_fail.search(line)
                        if m:
                            status[m.group(1)] = {"state": "failed"}
                            render_status()
                            continue
                        m = re_all.search(line)
                        if m:
                            status["__overall__"] = {"state": "done", "time": float(m.group(1))}
                            continue

                    proc.wait()
                    with log_box:
                        st.code("\n".join(log_text), language="text")

                    if proc.returncode == 0:
                        sac.alert(
                            title="Pipeline finished successfully",
                            description="All tasks completed",
                            status='success',
                            variant='fill',
                        )
                        with open(output_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        overall = status.pop("__overall__", None)
                        if overall:
                            status["__total__"] = overall
                        st.session_state.results = results
                        st.session_state.times = status
                    else:
                        st.error(f"Pipeline exited with code {proc.returncode}.")
                except FileNotFoundError:
                    st.error(f"Python interpreter not found: {sys.executable}")

    if "results" in st.session_state and "times" in st.session_state:
        show_results(st.session_state.results, st.session_state.times)
