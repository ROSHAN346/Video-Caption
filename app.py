"""Professional Streamlit UI — Video Captioning Pipeline."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

try:
    import streamlit_antd_components as sac
    SAC_AVAILABLE = True
except ImportError:
    sac = None
    SAC_AVAILABLE = False

try:
    import config as cfg
except Exception:
    cfg = None


def _auto_scroll_js() -> str:
    """Returns JS that scrolls the page to the pipeline-progress-anchor element."""
    return """
    <script>
        (function() {
            const tries = [50, 200, 500, 1000, 1500];
            tries.forEach(function(delay) {
                setTimeout(function() {
                    const el = document.getElementById('pipeline-progress-anchor');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        window.scrollBy({ top: -80, behavior: 'smooth' });
                    }
                }, delay);
            });
        })();
    </script>
    """


def _video_short_name(video_path: str) -> str:
    stem = Path(video_path).stem
    prefix = stem.split("-", 1)[0]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", prefix).strip("._-")
    return safe or stem


# ===========================================================================
# Page Config
# ===========================================================================
st.set_page_config(
    page_title="No-Cap AI — Video Captioning",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Theme CSS — Professional Dark
# ===========================================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>

        :root {
            --bg: #0a0e14;
            --surface-1: #11161d;
            --surface-2: #161c25;
            --surface-3: #1c2330;
            --border: #232a36;
            --border-strong: #2d3543;
            --text: #e6edf3;
            --text-secondary: #9ba6b4;
            --text-muted: #6b7684;
            --accent: #4f8cff;
            --accent-hover: #6aa0ff;
            --accent-subtle: rgba(79, 140, 255, 0.12);
            --accent-border: rgba(79, 140, 255, 0.35);
            --success: #4ade80;
            --success-subtle: rgba(74, 222, 128, 0.12);
            --error: #ef4444;
            --error-subtle: rgba(239, 68, 68, 0.12);
            --warning: #fbbf24;
            --warning-subtle: rgba(251, 191, 36, 0.12);
        }

        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* -------- Layout -------- */
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
            max-width: 1100px;
        }

        .main .block-container { background: var(--bg); }

        /* -------- Page Header -------- */
        .page-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.25rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }
        .page-header-left { display: flex; align-items: center; gap: 0.7rem; }
        .page-header-icon {
            width: 34px; height: 34px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem;
        }
        .page-header-title { font-size: 1.15rem; font-weight: 600; color: var(--text); letter-spacing: -0.2px; }
        .page-header-title small { color: var(--text-muted); font-weight: 400; margin-left: 0.5rem; font-size: 0.78rem; }
        .page-header-status {
            display: inline-flex; align-items: center; gap: 0.4rem;
            color: var(--success);
            font-size: 0.78rem; font-weight: 500;
        }
        .page-header-status::before {
            content: ''; width: 6px; height: 6px; border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        /* -------- Cards -------- */
        .card {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            transition: border-color 0.15s ease;
        }
        .card:hover { border-color: var(--border-strong); }
        .card-elevated {
            background: var(--surface-2);
            border: 1px solid var(--border-strong);
            border-radius: 10px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }
        .card-header {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        .card-title {
            font-size: 0.95rem; font-weight: 600;
            color: var(--text);
            display: flex; align-items: center; gap: 0.5rem;
            margin: 0;
        }
        .card-subtitle { font-size: 0.82rem; color: var(--text-muted); }

        /* -------- Stats -------- */
        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
        .stat-card {
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }
        .stat-icon { font-size: 0.9rem; margin-bottom: 0.4rem; }
        .stat-value {
            font-size: 1.5rem; font-weight: 700; color: var(--text);
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }
        .stat-label {
            font-size: 0.72rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.6px;
            font-weight: 500;
            margin-top: 0.35rem;
        }

        /* -------- Section Headers -------- */
        .section-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin: 1.5rem 0 0.75rem 0;
        }
        .divider { height: 1px; background: var(--border); margin: 1.5rem 0; border: none; }
        .gap-sm { margin-top: 0.75rem; }
        .gap-md { margin-top: 1.25rem; }
        .gap-lg { margin-top: 2rem; }

        /* -------- Captions -------- */
        .caption-card {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            color: var(--text);
            line-height: 1.7;
            font-size: 0.93rem;
        }

        /* -------- Keyframes -------- */
        .kf-card {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem;
            text-align: center;
        }
        .kf-meta { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.4rem; }
        .kf-time { color: var(--accent); font-weight: 600; font-size: 0.85rem; }

        /* -------- Status Badge -------- */
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 500;
            border: 1px solid;
        }
        .status-pill-success { background: var(--success-subtle); color: var(--success); border-color: rgba(74, 222, 128, 0.3); }
        .status-pill-error   { background: var(--error-subtle); color: var(--error); border-color: rgba(239, 68, 68, 0.3); }
        .status-pill-info    { background: var(--accent-subtle); color: var(--accent); border-color: var(--accent-border); }
        .status-pill-warning { background: var(--warning-subtle); color: var(--warning); border-color: rgba(251, 191, 36, 0.3); }

        /* -------- Sidebar -------- */
        [data-testid="stSidebar"] {
            background: var(--surface-1);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        .sidebar-header {
            display: flex; align-items: center; gap: 0.6rem;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        .sidebar-icon {
            width: 28px; height: 28px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 7px;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.95rem;
        }
        .sidebar-title { font-size: 1rem; font-weight: 600; color: var(--text); }
        .sidebar-desc { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1.25rem; line-height: 1.5; }

        .info-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.78rem;
        }
        .info-row:last-child { border-bottom: none; }
        .info-key { color: var(--text-muted); }
        .info-val { color: var(--text); font-family: 'SF Mono', JetBrains Mono, monospace; font-size: 0.72rem; }

        /* -------- Tabs -------- */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            border-bottom: 1px solid var(--border);
            border-radius: 0;
            padding: 0;
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 0;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.7rem 1.25rem;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }
        .stTabs [data-baseweb="tab"]:hover { color: var(--text-secondary); }
        .stTabs [aria-selected="true"] {
            color: var(--text);
            background: transparent;
            border-bottom: 2px solid var(--accent);
        }
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

        /* -------- Buttons -------- */
        .stButton > button {
            background: var(--accent) !important;
            color: white !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.25rem !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:hover {
            background: var(--accent-hover) !important;
            box-shadow: 0 4px 12px rgba(79, 140, 255, 0.3) !important;
        }

        /* -------- File Uploader -------- */
        [data-testid="stFileUploader"] section {
            padding: 1rem 1.25rem !important;
        }
        .stFileUploader [data-baseweb="file-uploader"] {
            background: var(--surface-1);
            border-radius: 10px;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface-1) !important;
            border: 1px dashed var(--border-strong) !important;
            border-radius: 10px !important;
            padding: 1.5rem !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent) !important;
            background: var(--surface-2) !important;
        }

        /* -------- Inputs -------- */
        .stTextInput > div > div {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .stTextInput > div > div:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-subtle);
        }
        .stSelectbox > div > div {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .stTextArea textarea {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        /* -------- Expander -------- */
        details {
            background: var(--surface-1) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        summary {
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            color: var(--text) !important;
        }

        /* -------- Code -------- */
        .stCodeBlock {
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        code { font-family: 'SF Mono', JetBrains Mono, monospace !important; font-size: 0.78rem !important; }

        /* -------- Video (Standard 16:9) -------- */
        video {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            width: 100% !important;
            aspect-ratio: 16 / 9 !important;
            object-fit: contain !important;
            background: #000 !important;
        }

        /* -------- Tables -------- */
        .stTable { border-radius: 8px; overflow: hidden; }
        table { border-collapse: collapse !important; }
        th { background: var(--surface-2) !important; color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.78rem !important; }
        td { color: var(--text) !important; font-size: 0.83rem !important; }

        /* -------- Alerts -------- */
        .stAlert {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
            background: var(--surface-2) !important;
        }

        /* -------- Hide Streamlit Stuff -------- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* -------- Scrollbar -------- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _status_pill(text: str, kind: str = "info") -> str:
    return f'<span class="status-pill status-pill-{kind}">{text}</span>'


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-icon">🎬</div>
            <div>
                <div class="sidebar-title">No-Cap AI</div>
            </div>
        </div>
        <p class="sidebar-desc">AI-powered video captioning — scene detection, keyframe extraction & multi-style generation.</p>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        api_key_input = st.text_input(
            "Fireworks API Key",
            value=os.getenv("FIREWORKS_API_KEY", ""),
            type="password",
            help="Optional. Direct API key if not using default backend.",
        )
        worker_url_input = os.getenv("WORKER_URL", "https://patient-violet-5828.viratforedu175.workers.dev")
        if worker_url_input:
            os.environ["WORKER_URL"] = worker_url_input
        if api_key_input:
            os.environ["FIREWORKS_API_KEY"] = api_key_input
            os.environ["FIREWORKS_TEXT_API_KEY"] = api_key_input

    if cfg is not None:
        with st.expander("Model & Parameters", expanded=False):
            rows = [
                ("Vision Model", getattr(cfg, "FIREWORKS_VISION_MODEL", "")),
                ("Text Model", getattr(cfg, "FIREWORKS_TEXT_MODEL", "")),
            ]
            for k, v in rows:
                st.markdown(
                    f'<div class="info-row"><span class="info-key">{k}</span>'
                    f'<span class="info-val">{v.split("/")[-1] if v else "—"}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            params = [
                ("Max Frames", str(getattr(cfg, "MAX_FRAMES", ""))),
                ("Candidate FPS", str(getattr(cfg, "CANDIDATE_FPS", ""))),
                ("Batch Size", str(getattr(cfg, "EMBEDDING_BATCH_SIZE", ""))),
            ]
            for k, v in params:
                st.markdown(
                    f'<div class="info-row"><span class="info-key">{k}</span>'
                    f'<span class="info-val">{v}</span></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-row"><span class="info-key">Python</span>'
        f'<span class="info-val">{sys.version_info.major}.{sys.version_info.minor}</span></div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page Header
# ===========================================================================
st.markdown(
    """
    <div class="page-header">
        <div class="page-header-left">
            <div class="page-header-icon">🎬</div>
            <div>
                <div class="page-header-title">No-Cap AI <small>Video Captioning Pipeline</small></div>
            </div>
        </div>
        <div class="page-header-status">Online</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Results (Batch Mode)
# ===========================================================================
def show_results(results: list, times: dict):
    if not results:
        return

    task_times = {k: v for k, v in times.items() if not k.startswith("__")}
    failed = [tid for tid, v in task_times.items() if v.get("state") == "failed"]
    overall = times.get("__total__", {}).get("time")
    total_time = overall if overall is not None else sum(v["time"] for v in task_times.values() if v.get("state") == "done")
    avg_time = (total_time / len(task_times)) if task_times else 0.0

    # Stats row
    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-icon">🎞</div>
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏱</div>
                <div class="stat-value">{total_time:.1f}s</div>
                <div class="stat-label">Total Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">{avg_time:.1f}s</div>
                <div class="stat-label">Avg / Video</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">{"✅" if not failed else "⚠️"}</div>
                <div class="stat-value">{len(failed)}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Per-video
    task_ids = [r.get("task_id", f"task_{i}") for i, r in enumerate(results)]
    # Use a stable hash of the task_ids as key to avoid StreamlitDuplicateElementId when re-rendered
    key_hash = hashlib.md5(",".join(task_ids).encode()).hexdigest()[:10]
    selected = st.selectbox(
        "Select video",
        task_ids,
        index=0,
        label_visibility="collapsed",
        key=f"batch_result_select_{key_hash}",
    )

    idx = task_ids.index(selected)
    result = results[idx]

    # Card header with status
    state_html = ""
    if selected in times:
        t = times[selected]
        if t.get("state") == "done":
            state_html = _status_pill(f"Completed in {t['time']:.1f}s", "success")
        elif t.get("state") == "failed":
            state_html = _status_pill("Failed", "error")

    st.markdown(
        f"""
        <div class="card-elevated">
            <div class="card-header">
                <div class="card-title">📁 {selected}</div>
                {state_html}
            </div>
        """,
        unsafe_allow_html=True,
    )

    captions = result.get("captions", {})
    if not captions:
        st.markdown('<p style="color:var(--text-muted);font-size:0.88rem;">No captions generated.</p>', unsafe_allow_html=True)
    else:
        style_names = list(captions.keys())
        tab_objs = st.tabs([s.replace("_", " ").title() for s in style_names])
        for tab_obj, style in zip(tab_objs, style_names):
            with tab_obj:
                text = captions[style]
                if isinstance(text, str) and text.strip():
                    st.markdown(f'<div class="caption-card">{text}</div>', unsafe_allow_html=True)
                else:
                    st.json(text if text else {"empty": True})

    st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# Tabs
# ===========================================================================
tab_single, tab_batch = st.tabs(["Single Video", "Batch Processing"])


# ===========================================================================
# Tab 1 — Single Video
# ===========================================================================
with tab_single:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Max file size ~500MB",
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        col1, col2 = st.columns([1.4, 1], gap="medium")
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-header"><div class="card-title">📹 Input Video</div>'
                f'<span class="card-subtitle">{uploaded_video.name}</span></div>',
                unsafe_allow_html=True,
            )
            st.video(uploaded_video)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="card-header"><div class="card-title">⚙️ Configuration</div></div>',
                unsafe_allow_html=True,
            )
            style_choices = ["All Styles", "formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
            selected_style = st.selectbox(
                "Caption Tone",
                style_choices,
                index=0,
                help="'All Styles' generates captions in 4 different tones.",
                key="single_style_select",
            )
            st.markdown('<div class="gap-sm"></div>', unsafe_allow_html=True)
            run_btn = st.button("▶  Run Pipeline", type="primary", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

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

            progress_anchor = st.empty()
            log_lines = []

            def render_progress(stage: int, msg: str, status: str = "running"):
                kind = "info" if status == "running" else ("success" if status == "done" else "error")
                status_text = "Running" if status == "running" else ("Complete" if status == "done" else "Error")
                with progress_anchor.container():
                    st.iframe(_auto_scroll_js(), height=0)
                    st.markdown(
                        f"""
                        <div id="pipeline-progress-anchor"></div>
                        <div class="card-elevated" style="border-color: var(--accent-border);">
                            <div class="card-header">
                                <div class="card-title">
                                    <span style="
                                        display:inline-flex;
                                        width:22px;height:22px;
                                        background:var(--accent-subtle);
                                        border-radius:6px;
                                        align-items:center;justify-content:center;
                                        margin-right:6px;
                                    ">⚡</span>
                                    Pipeline Progress
                                </div>
                                <span class="status-pill status-pill-{kind}">{status_text}</span>
                            </div>
                            <p style="color:var(--text);font-size:0.95rem;font-weight:500;margin:0 0 1rem 0;">
                                {msg}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    step_titles = ["Scene Detection", "Keyframe Selection", "Vision Analysis", "Caption Generation"]
                    step_descs = ["Split video into scenes", "Extract representative frames", "Analyze frames with AI", "Generate styled captions"]
                    if SAC_AVAILABLE:
                        sac.steps(
                            items=[sac.StepsItem(title=t, description=d) for t, d in zip(step_titles, step_descs)],
                            index=stage if status == "running" else 4,
                            size='sm',
                            direction='horizontal',
                            key=f"progress_step_{stage}_{status}",
                        )
                    else:
                        # Fallback: show progress as numbered list
                        st.markdown(
                            "".join(
                                f'<div style="padding:0.4rem 0;color:var(--text-primary,{""});font-size:0.85rem;'
                                f'{"opacity:1;color:var(--accent);" if i == stage or (status == "done" and i == 3) else "opacity:0.5;"}'
                                f'border-left:2px solid {"var(--accent)" if i == stage or (status == "done" and i == 3) else "var(--border)"};'
                                f'padding-left:0.75rem;margin:0.2rem 0;">'
                                f'{"✓ " if (status == "done" and i < 4) or (status == "running" and i < stage) else ""}'
                                f'{i+1}. {t}</div>'
                                for i, (t, d) in enumerate(zip(step_titles, step_descs))
                            ),
                            unsafe_allow_html=True,
                        )

            render_progress(0, "Initializing pipeline...", "running")

            try:
                sub_env = os.environ.copy()
                if worker_url_input:
                    sub_env["WORKER_URL"] = worker_url_input
                if api_key_input:
                    sub_env["FIREWORKS_API_KEY"] = api_key_input
                    sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=sub_env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                current_step = 0
                for raw in proc.stdout:
                    line = raw.strip() if raw else ""
                    if not line:
                        continue
                    log_lines.append(line)

                    new_step = current_step
                    new_msg = "Processing..."

                    if "Scene Detection" in line or "detect_scenes" in line:
                        new_step, new_msg = 1, "Detecting scenes & capturing frame candidates"
                    elif "select_keyframes" in line or "Selected keyframes" in line:
                        new_step, new_msg = 2, "Selecting representative keyframes"
                    elif "Analyzing" in line and "vision" in line:
                        new_step, new_msg = 3, "Analyzing keyframes with Fireworks vision"
                    elif "Generating" in line and ("styles" in line or "report" in line.lower()):
                        new_step, new_msg = 4, "Generating multi-style captions"

                    if new_step != current_step:
                        current_step = new_step
                        render_progress(new_step, new_msg, "running")

                proc.wait()
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

                if proc.returncode == 0:
                    render_progress(4, "All captions generated successfully!", "done")
                    output_dir_path = Path("output") / short_name
                    captions_file = output_dir_path / "captions.json"
                    if captions_file.exists():
                        with open(captions_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        st.session_state["single_result"] = data
                        st.session_state["single_short_name"] = short_name
                    else:
                        st.error("Output captions.json was not created.")
                else:
                    render_progress(current_step, f"Pipeline failed (exit code {proc.returncode})", "error")
            except Exception as e:
                st.error(f"Error: {e}")

        # Results section
        if "single_result" in st.session_state:
            data = st.session_state["single_result"]
            short_name = st.session_state["single_short_name"]

            st.markdown('<div class="section-title">📊 Video Analysis Results</div>', unsafe_allow_html=True)

            captions = data.get("captions", {})
            if not captions:
                st.markdown(
                    '<div class="card"><p style="color:var(--text-muted);">No captions generated.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                style_names = list(captions.keys())
                tabs = st.tabs([s.replace("_", " ").title() for s in style_names])
                for tab, (style, text) in zip(tabs, captions.items()):
                    with tab:
                        st.markdown(
                            f'<div class="caption-card">{text}</div>',
                            unsafe_allow_html=True,
                        )

            # Keyframes
            keyframes = data.get("keyframes", [])
            if keyframes:
                st.markdown('<div class="section-title">🖼️ Keyframe Selection</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="color:var(--text-muted);font-size:0.83rem;margin-bottom:0.85rem;">'
                    'Representative frames extracted from the video.</p>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(min(len(keyframes), 5))
                for i, kf in enumerate(keyframes):
                    img_path = Path("output") / short_name / kf.get("image_path", "")
                    with cols[i % len(cols)]:
                        st.markdown('<div class="kf-card">', unsafe_allow_html=True)
                        if img_path.exists():
                            st.image(str(img_path), width='stretch')
                        st.markdown(
                            f'<div class="kf-meta">Scene {kf.get("scene_id")}</div>'
                            f'<div class="kf-time">{kf.get("timestamp_sec")}s</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

            # Scenes timeline
            scenes = data.get("scenes", [])
            if scenes:
                st.markdown('<div class="section-title">🎬 Detected Scenes</div>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.table(scenes)
                st.markdown('</div>', unsafe_allow_html=True)

            # Download
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.download_button(
                "Download captions.json",
                data=json.dumps(data, indent=2),
                file_name="captions.json",
                mime="application/json",
            )


# ===========================================================================
# Tab 2 — Batch Processing
# ===========================================================================
with tab_batch:
    uploaded_batch = st.file_uploader(
        "Upload tasks JSON",
        type=["json"],
        key="batch_uploader",
        label_visibility="collapsed",
    )

    if uploaded_batch is not None:
        try:
            tasks = json.loads(uploaded_batch.getvalue().decode("utf-8"))
            if not isinstance(tasks, list):
                st.error("Top-level JSON must be a list of tasks.")
                st.stop()

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card-header"><div class="card-title">📋 Loaded Tasks</div>'
                f'{_status_pill(f"{len(tasks)} task(s) ready", "success")}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Preview JSON"):
                st.json(tasks)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="card-elevated"><div class="card-header">'
                '<div class="card-title">🚀 Execute Batch</div></div></div>',
                unsafe_allow_html=True,
            )
            run_col, _ = st.columns([1, 3])
            run_batch = run_col.button(
                "▶  Run Batch Pipeline",
                type="primary",
                width='stretch',
            )

            if run_batch:
                with tempfile.TemporaryDirectory() as tmp:
                    input_path = os.path.join(tmp, "tasks.json")
                    with open(input_path, "w", encoding="utf-8") as f:
                        f.write(uploaded_batch.getvalue().decode("utf-8"))
                    output_path = os.path.join(tmp, "results.json")
                    cmd = [sys.executable, "main.py", "--input", input_path, "--output", output_path]

                    status = {}
                    status_box = st.empty()
                    log_exp = st.expander("Process Log", expanded=False)
                    log_text = []

                    def render():
                        if not status:
                            status_box.info("Waiting for tasks to start...")
                            return
                        items_html = ""
                        for tid, info in status.items():
                            if info["state"] == "running":
                                items_html += f'<div class="info-row"><span class="info-key">{tid}</span>{_status_pill("Processing", "info")}</div>'
                            elif info["state"] == "done":
                                t_disp = f"{info['time']:.1f}s"
                                items_html += f'<div class="info-row"><span class="info-key">{tid}</span>{_status_pill("Done in " + t_disp, "success")}</div>'
                            else:
                                items_html += f'<div class="info-row"><span class="info-key">{tid}</span>{_status_pill("Failed", "error")}</div>'
                        status_box.markdown(f'<div class="card">{items_html}</div>', unsafe_allow_html=True)

                    render()
                    log_container = log_exp.empty()

                    try:
                        sub_env = os.environ.copy()
                        if worker_url_input:
                            sub_env["WORKER_URL"] = worker_url_input
                        if api_key_input:
                            sub_env["FIREWORKS_API_KEY"] = api_key_input
                            sub_env["FIREWORKS_TEXT_API_KEY"] = api_key_input

                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=sub_env,
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
                            log_container.code("\n".join(log_text[-25:]), language="text")

                            m = re_start.search(line)
                            if m:
                                status[m.group(1)] = {"state": "running"}
                                render()
                                continue
                            m = re_done.search(line)
                            if m:
                                status[m.group(1)] = {"state": "done", "time": float(m.group(2))}
                                render()
                                continue
                            m = re_fail.search(line)
                            if m:
                                status[m.group(1)] = {"state": "failed"}
                                render()
                                continue
                            m = re_all.search(line)
                            if m:
                                status["__overall__"] = {"state": "done", "time": float(m.group(1))}
                                continue

                        proc.wait()
                        if proc.returncode == 0:
                            st.markdown(
                                f'<div class="card" style="border-left:3px solid var(--success);">'
                                f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                                f'<span style="color:var(--success);font-size:1.1rem;">✅</span>'
                                f'<span style="color:var(--text);font-size:0.9rem;font-weight:500;">Pipeline finished successfully</span></div></div>',
                                unsafe_allow_html=True,
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

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    if "results" in st.session_state and "times" in st.session_state:
        show_results(st.session_state.results, st.session_state.times)
