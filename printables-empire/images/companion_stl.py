"""Generate companion STL files for content entries.

Printables requires at least one model file per entry.
This generates a small, useful 3D-printable companion piece
relevant to each content type.
"""

from pathlib import Path

import numpy as np
import trimesh


def create_test_block(output_path: str, label: str = "ForgeFiles") -> str:
    """Create a small rectangular test block STL.

    A simple 40x20x5mm block that can serve as a print settings
    test piece or display stand companion. Useful with any article
    since users can verify their print settings with it.

    Args:
        output_path: Where to save the STL file.
        label: Not embossed (would need complex geometry), just for naming.

    Returns:
        Path to generated STL.
    """
    # Simple box: 40mm x 20mm x 5mm
    box = trimesh.creation.box(extents=[40, 20, 5])
    # Center on print bed (z=0 at bottom)
    box.apply_translation([0, 0, 2.5])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    box.export(output_path, file_type="stl")
    return output_path


def create_layer_test(output_path: str) -> str:
    """Create a layer height test piece — stepped pyramid.

    A small stepped block with 4 heights (2mm, 4mm, 6mm, 8mm)
    so users can see how different layer heights look on the
    same print. 30x30mm footprint.
    """
    meshes = []
    step_width = 7.5  # 4 steps across 30mm

    for i, height in enumerate([2, 4, 6, 8]):
        step = trimesh.creation.box(extents=[step_width, 30, height])
        x_offset = -11.25 + (i * step_width)
        step.apply_translation([x_offset, 0, height / 2])
        meshes.append(step)

    combined = trimesh.util.concatenate(meshes)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, file_type="stl")
    return output_path


def get_companion_stl(content_type: str, output_dir: str, topic: str = "") -> str:
    """Get the appropriate companion STL for a content type.

    Args:
        content_type: article, review, listing, or post.
        output_dir: Directory to save the STL.
        topic: Topic string to determine which companion to use.

    Returns:
        Path to the STL file.
    """
    output_path = str(Path(output_dir) / "companion.stl")

    if "layer" in topic.lower() or "height" in topic.lower():
        return create_layer_test(output_path)

    return create_test_block(output_path)
