"""
Report Cache Service

Provides file-based caching for generated reports to avoid
regeneration for already-processed scenes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReportCache:
    """File-based report cache."""

    def __init__(self, base_dir: str = "reports"):
        """
        Initialize cache.

        Args:
            base_dir: Base directory for cached reports
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, scene_id: int, style: str) -> Path:
        """
        Get the file path for a cached report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Path to cached report file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / f"{style}.md"

    def get_pdf_path(self, scene_id: int, style: str) -> Path:
        """
        Get the file path for a cached PDF report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Path to cached PDF file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / f"{style}.pdf"

    def get_cache_metadata_path(self, scene_id: int) -> Path:
        """
        Get the file path for cache metadata.

        Args:
            scene_id: Scene ID

        Returns:
            Path to metadata file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / "cache.json"

    def is_cached(self, scene_id: int, style: str) -> bool:
        """
        Check if a report is cached.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            True if report exists in cache
        """
        cache_path = self.get_cache_path(scene_id, style)
        return cache_path.exists()

    def get_cached_report(self, scene_id: int, style: str) -> Optional[str]:
        """
        Retrieve a cached report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Cached report text or None
        """
        cache_path = self.get_cache_path(scene_id, style)

        if not cache_path.exists():
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    def cache_report(
        self,
        scene_id: int,
        style: str,
        report: str,
        metadata: dict = None
    ):
        """
        Cache a generated report.

        Args:
            scene_id: Scene ID
            style: Report style
            report: Report text
            metadata: Optional metadata to store
        """
        # Save report
        cache_path = self.get_cache_path(scene_id, style)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Update metadata
        self._update_metadata(scene_id, style, metadata)

        logger.info(f"Cached {style} report for scene {scene_id}")

    def _update_metadata(self, scene_id: int, style: str, metadata: dict = None):
        """Update cache metadata."""
        meta_path = self.get_cache_metadata_path(scene_id)

        # Load existing metadata
        existing = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                existing = json.load(f)

        # Update
        existing["scene_id"] = scene_id
        existing["generated_at"] = datetime.now().isoformat()

        if "styles" not in existing:
            existing["styles"] = []

        if style not in existing["styles"]:
            existing["styles"].append(style)

        if metadata:
            existing.update(metadata)

        # Save
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)

    def clear_cache(self, scene_id: int = None):
        """
        Clear cache for a specific scene or all scenes.

        Args:
            scene_id: Scene ID to clear (None = clear all)
        """
        import shutil

        if scene_id is not None:
            scene_dir = self.base_dir / f"scene_{scene_id:03d}"
            if scene_dir.exists():
                shutil.rmtree(scene_dir)
                logger.info(f"Cleared cache for scene {scene_id}")
        else:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all report cache")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_reports = 0
        total_scenes = 0
        total_size = 0

        if self.base_dir.exists():
            for scene_dir in self.base_dir.iterdir():
                if scene_dir.is_dir() and scene_dir.name.startswith("scene_"):
                    total_scenes += 1
                    for report_file in scene_dir.glob("*.md"):
                        total_reports += 1
                        total_size += report_file.stat().st_size

        return {
            "total_scenes": total_scenes,
            "total_reports": total_reports,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
