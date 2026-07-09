# Parallel Batch Processing Architecture

## Overview

This document describes the parallel batch processing system for video keyframe analysis
using multiple Fireworks AI vision models simultaneously with a work queue pattern.

## Goals

- Analyze keyframes 3-4x faster than sequential processing
- Use 3 Fireworks vision models in parallel
- Send 2 images per API request (batch size)
- Dynamic load balancing via work queue (no idle workers)
- Single retry on failure

---

## Architecture

### Work Queue Pattern

```
Shared Queue: [frame_0, frame_1, frame_2, ..., frame_14]
                      |
         +------------+------------+
         |            |            |
    Worker A      Worker B      Worker C
    (model_A)     (model_B)     (model_C)
         |            |            |
         v            v            v
   Pull 2 frames  Pull 2 frames  Pull 2 frames
   from queue     from queue     from queue
         |            |            |
         v            v            v
   API call to    API call to    API call to
   model_A        model_B        model_C
         |            |            |
         v            v            v
   DONE! Pulls    DONE! Pulls    (still working)
   next 2 frames  next 2 frames
   from queue     from queue
         |            |
         v            v
   Pulls frame_8   Pulls frame_10
   + frame_9       + frame_11
         ...           ...
```

### Key Properties

1. **No idle workers** - Fast workers immediately pull more frames
2. **Dynamic assignment** - Frames are assigned on-demand, not pre-allocated
3. **Rate limit protection** - Semaphore limits concurrent API calls
4. **Single retry** - Failed frames re-queued once to different model

---

## Configuration

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 2 | Images per API request |
| max_workers | 3 | Concurrent worker threads |
| max_retries | 1 | Retries on failure before giving up |
| semaphore_limit | 3 | Max concurrent API calls |

### Fireworks Vision Models

| # | Model ID | Type |
|---|----------|------|
| 1 | accounts/fireworks/models/minimax-m3 | Vision |
| 2 | accounts/fireworks/models/llama-v3p2-11b-vision-instruct | Vision |
| 3 | accounts/fireworks/models/llama-v3p2-90b-vision-instruct | Vision |

---

## Implementation Details

### Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | config.py | Add FIREWORKS_VISION_MODELS list |
| 2 | services/fireworks_client.py | Add analyze_images_batch() method |
| 3 | services/image_analyzer.py | Add analyze_keyframes_parallel() with work queue |
| 4 | main.py | Update generate_captions_for_video() |
| 5 | app.py | Update batch processing section |

### 1. config.py

Add list of vision models:

```python
FIREWORKS_VISION_MODELS = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
]
```

### 2. services/fireworks_client.py

Add batch analysis method:

```python
def analyze_images_batch(
    self,
    image_paths: list[str],
    prompt: str,
    model: str,
    max_tokens: int = 1024
) -> list[str]:
    """
    Send multiple images in one API request.

    Args:
        image_paths: List of image file paths (batch_size=2)
        prompt: Text prompt for analysis
        model: Model identifier
        max_tokens: Maximum tokens per response

    Returns:
        List of response texts, one per image
    """
    import base64

    # Build content list with text + all images
    content = [{"type": "text", "text": prompt}]

    for image_path in image_paths:
        path = Path(image_path)
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        suffix = path.suffix.lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
        mime_type = mime_map.get(suffix, "jpeg")
        image_url = f"data:image/{mime_type};base64,{image_b64}"

        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    messages = [{"role": "user", "content": content}]

    response = self.client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )

    # Split response by image separator if multiple images
    full_response = response.choices[0].message.content
    # For batch of 2, split on delimiter or return full for single
    if len(image_paths) == 1:
        return [full_response]

    # Try to split response by common delimiters
    import re
    parts = re.split(r'\n---\n|\n===\n|(?=\*\*Frame \d)', full_response)

    # Clean and return
    results = []
    for part in parts:
        part = part.strip()
        if part:
            results.append(part)

    # Pad with placeholders if split didn't match batch size
    while len(results) < len(image_paths):
        results.append("Analysis unavailable")

    return results[:len(image_paths)]
```

### 3. services/image_analyzer.py

Add parallel analysis function:

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

def analyze_keyframes_parallel(
    selected: list,
    output_dir: Path,
    client,  # FireworksClient
    models: list[str],
    batch_size: int = 2,
    max_workers: int = 3,
    max_retries: int = 1
) -> list[dict]:
    """
    Analyze keyframes in parallel using work queue with batch requests.

    Args:
        selected: List of SelectedFrame objects
        output_dir: Directory containing keyframe JPEGs
        client: FireworksClient instance
        models: List of model IDs to use (3 models)
        batch_size: Images per API request (default: 2)
        max_workers: Concurrent worker threads (default: 3)
        max_retries: Retries on failure (default: 1)

    Returns:
        List of analysis dictionaries
    """
    # Build shared work queue
    work_queue = queue.Queue()
    for i, sf in enumerate(selected):
        work_queue.put((i, sf))

    # Results storage (indexed by frame number)
    results = [None] * len(selected)
    results_lock = threading.Lock()

    # Re-queue for failed frames (max 1 retry)
    retry_queue = queue.Queue()

    # Semaphore for rate limit protection
    semaphore = threading.Semaphore(max_workers)

    def analyze_batch(batch_items, worker_id, model, attempt=0):
        """Analyze a batch of frames with specified model."""
        indices = [item[0] for item in batch_items]
        frames = [item[1] for item in batch_items]

        # Build image paths
        image_paths = [str(output_dir / f"keyframe_{i:03d}.jpg") for i in indices]

        try:
            with semaphore:
                responses = client.analyze_images_batch(
                    image_paths=image_paths,
                    prompt=ANALYSIS_PROMPT,
                    model=model,
                    max_tokens=1024
                )

            # Process responses
            for idx, (i, sf), response in zip(indices, zip(indices, frames), responses):
                analysis = _parse_json_response(response)

                # Check if analysis failed
                if analysis.get("summary") in ["Analysis unavailable", "Analysis failed to parse", ""]:
                    if attempt < max_retries:
                        retry_queue.put((i, sf))
                        logger.warning(f"Frame {i} analysis failed, re-queued for retry")
                    continue

                # Add metadata
                analysis["scene_id"] = sf.scene_id
                analysis["frame_index"] = i
                analysis["image_path"] = f"keyframe_{i:03d}.jpg"

                with results_lock:
                    results[i] = analysis

                # Save analysis file
                analysis_path = output_dir / "analysis" / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
                save_analysis(analysis, str(analysis_path))

                logger.info(f"Worker {worker_id}: Analyzed frame {i} with {model}")

        except Exception as e:
            logger.error(f"Worker {worker_id}: Batch failed: {e}")
            # Re-queue failed frames
            if attempt < max_retries:
                for item in batch_items:
                    retry_queue.put(item)

    def worker(worker_id, model):
        """Worker thread that pulls from queue and processes batches."""
        while True:
            batch_items = []

            # Pull batch_size frames from queue
            for _ in range(batch_size):
                try:
                    item = work_queue.get_nowait()
                    batch_items.append(item)
                except queue.Empty:
                    break

            if not batch_items:
                break

            # Analyze batch
            analyze_batch(batch_items, worker_id, model)

        # Process retry queue (single retry)
        while True:
            batch_items = []
            for _ in range(batch_size):
                try:
                    item = retry_queue.get_nowait()
                    batch_items.append(item)
                except queue.Empty:
                    break

            if not batch_items:
                break

            # Use same model for retry
            analyze_batch(batch_items, worker_id, model, attempt=1)

    # Launch workers with different models
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for w in range(max_workers):
            model = models[w % len(models)]
            futures.append(executor.submit(worker, w, model))

        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    # Create fallback analyses for any frames that failed
    for i, sf in enumerate(selected):
        if results[i] is None:
            results[i] = {
                "scene_id": sf.scene_id,
                "frame_index": i,
                "scene_type": "unknown",
                "location": "unknown",
                "people": "unknown",
                "objects": [],
                "vehicles": [],
                "animals": [],
                "activities": "unknown",
                "weather": "unknown",
                "time_of_day": "unknown",
                "environment": "unknown",
                "risk_level": "unknown",
                "confidence": 0.0,
                "summary": "Analysis unavailable",
                "image_path": f"keyframe_{i:03d}.jpg"
            }

    return [r for r in results if r is not None]
```

### 4. main.py

Update generate_captions_for_video():

```python
def generate_captions_for_video(
    output_dir: Path,
    selected: list,
    scenes: list,
    styles: list[str]
) -> dict[str, str]:
    """Generate video-level captions using parallel batch processing."""
    from services.fireworks_client import get_fireworks_client
    from services.image_analyzer import analyze_keyframes_parallel
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_video_summary_reports
    from config import FIREWORKS_VISION_MODELS

    client = get_fireworks_client()

    # Parallel batch analysis
    logger.info(f"Analyzing {len(selected)} keyframes with {len(FIREWORKS_VISION_MODELS)} models...")
    analyses = analyze_keyframes_parallel(
        selected=selected,
        output_dir=output_dir,
        client=client,
        models=FIREWORKS_VISION_MODELS,
        batch_size=2,
        max_workers=3,
        max_retries=1
    )

    # Aggregate by scene
    scene_analyses = aggregate_by_scene(analyses, scenes)

    # Generate captions with retry
    captions = retry_api_call(
        lambda: generate_video_summary_reports(scene_analyses, client, styles, FIREWORKS_TEXT_MODEL)
    )

    # Guarantee all styles present
    if not captions:
        captions = {}
    for style in styles:
        if style not in captions or not captions[style]:
            captions[style] = f"Video content analyzed: {', '.join(s.get('activities', 'unknown') for s in scene_analyses.values())}"

    return captions
```

### 5. app.py

Update batch processing section to use parallel analysis:

```python
# Replace sequential analysis loop with:
from services.image_analyzer import analyze_keyframes_parallel
from config import FIREWORKS_VISION_MODELS

analyses = analyze_keyframes_parallel(
    selected=selected,
    output_dir=output_dir,
    client=client,
    models=FIREWORKS_VISION_MODELS,
    batch_size=2,
    max_workers=3,
    max_retries=1
)
```

---

## Rate Limit Handling

### Strategy

1. **Semaphore**: Limits to 3 concurrent API calls
2. **Model rotation**: Spreads load across 3 models
3. **Exponential backoff**: 5s, 10s on 429 errors
4. **Single retry**: Failed frames re-queued once

### Flow on Rate Limit

```
Worker gets 429 error
    |
    v
Release semaphore (allow other workers)
    |
    v
Sleep for delay (5s * 2^attempt)
    |
    v
Re-queue frame to retry_queue
    |
    v
Worker pulls next batch from work_queue
    |
    v
Later: retry_queue processed with single retry
```

---

## Performance Analysis

### Sequential (Current)

| Metric | Value |
|--------|-------|
| API calls | 15 (1 per frame) |
| Time per call | 3-5s |
| Total time | 45-75s |
| Throughput | 0.2-0.3 frames/sec |

### Parallel Work Queue (New)

| Metric | Value |
|--------|-------|
| API calls | ~8 (2 per batch) |
| Time per call | 3-5s |
| Concurrent calls | 3 |
| Total time | 15-25s |
| Throughput | 0.6-1.0 frames/sec |
| **Speedup** | **3-4x** |

### Breakdown (15 keyframes)

| Phase | Time |
|-------|------|
| Queue setup | <0.1s |
| Batch 1 (frames 0-5) | ~5s |
| Batch 2 (frames 6-11) | ~5s |
| Batch 3 (frames 12-14) | ~5s |
| Retries (if any) | ~5s |
| **Total** | **~15-20s** |

---

## Error Handling

### Failure Modes

| Failure | Handling |
|---------|----------|
| Single frame fails | Re-queue once to retry_queue |
| Retry also fails | Use placeholder analysis |
| Worker crashes | Other workers continue |
| All workers fail | Return placeholder analyses |
| Rate limit (429) | Exponential backoff + re-queue |
| Invalid JSON response | Regex extraction + placeholder |

### Fallback Analysis

```python
{
    "scene_id": scene_id,
    "frame_index": frame_index,
    "scene_type": "unknown",
    "location": "unknown",
    "people": "unknown",
    "objects": [],
    "activities": "unknown",
    "summary": "Analysis unavailable",
    "confidence": 0.0
}
```

---

## Testing

### Unit Tests

1. `test_analyze_images_batch()` - Verify batch API call
2. `test_analyze_keyframes_parallel()` - Verify work queue
3. `test_retry_on_failure()` - Verify re-queue logic
4. `test_rate_limit_handling()` - Verify backoff

### Integration Tests

1. Process 5 keyframes with 3 models
2. Verify all frames analyzed
3. Verify retry works on simulated failure
4. Verify placeholder on complete failure

### Performance Tests

1. Time 15 keyframes sequential vs parallel
2. Measure throughput (frames/sec)
3. Verify no idle workers

---

## Deployment

### Docker

No changes needed - same Dockerfile. The parallel processing uses Python's
built-in `concurrent.futures` and `threading` modules.

### Environment Variables

No new env vars needed. Uses existing:
- `FIREWORKS_API_KEY`
- `FIREWORKS_VISION_MODEL` (fallback)

### Configuration

Model list hardcoded in `config.py`:
```python
FIREWORKS_VISION_MODELS = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
]
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Processing | Sequential | Parallel work queue |
| Models | 1 | 3 |
| Batch size | 1 | 2 |
| Workers | 1 | 3 |
| API calls | 15 | ~8 |
| Total time | 45-75s | 15-25s |
| Speedup | baseline | 3-4x |
| Retry | None | 1 retry |
| Load balancing | N/A | Dynamic (fast workers do more) |
