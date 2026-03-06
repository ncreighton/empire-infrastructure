"""Image Step — Generate branded images via article_images_pipeline."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict

from .base import BaseStep

log = logging.getLogger(__name__)

PIPELINE_SCRIPT = Path(__file__).parent.parent.parent.parent.parent / "article_images_pipeline.py"


class ImageStep(BaseStep):
    name = "image"
    description = "Generate branded images (blog_featured, Pinterest, social)"
    requires = []

    def execute(self, context: Dict) -> Dict:
        title = context["title"]
        site_slug = context["site_slug"]

        if not PIPELINE_SCRIPT.exists():
            return {"image_status": "skipped", "reason": "Pipeline script not found"}

        try:
            kwargs = dict(capture_output=True, text=True, timeout=120)
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                [sys.executable, str(PIPELINE_SCRIPT),
                 "--site", site_slug,
                 "--title", title,
                 "--enhanced"],
                **kwargs
            )

            if result.returncode == 0:
                return {
                    "image_status": "generated",
                    "image_output": result.stdout[:500],
                }
            else:
                return {
                    "image_status": "failed",
                    "image_error": result.stderr[:300],
                }
        except Exception as e:
            log.error(f"Image generation failed: {e}")
            return {"image_status": "failed", "image_error": str(e)}

    def dry_run(self, context: Dict) -> Dict:
        return {
            "step": self.name,
            "action": f"Generate 5 image types for '{context['title']}' on {context['site_slug']}",
            "script": str(PIPELINE_SCRIPT),
            "status": "dry_run",
        }
