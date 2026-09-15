"""
tripo_to_glb.py

Converts a single TripoSR output (mesh.obj + texture) into a normalized
.glb ready to be dropped into the Vinyasa Three.js renderer, and returns
the metadata needed to write a scene.json entry for it.

Normalization applied (mirrors what main.js currently hardcodes for the
sofa, but made generic/reusable):
  1. Recenter mesh horizontally (x, z) on its own bounding-box center.
  2. Drop mesh vertically so its lowest point sits at y = 0 (on the floor).
  3. Uniformly scale so its bounding-box width (x-axis) matches a target
     real-world width you supply (this is the piece TripoSR can't know
     on its own -- it has no sense of metric scale).

Usage:
    python tripo_to_glb.py \
        --obj output/0/mesh.obj \
        --out public/model/object_01.glb \
        --target-width 2.8
"""

import argparse
import json
import sys
from pathlib import Path

import trimesh
import numpy as np


def load_as_scene(obj_path: Path) -> trimesh.Scene:
    """Load the OBJ, always as a Scene, so materials/textures come along
    and multi-mesh exports are structurally preserved into the GLB."""
    loaded = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    return loaded


def normalize(scene: trimesh.Scene, target_width: float) -> dict:
    """Recenter (x,z), floor-align (y), and scale to target_width in place.
    Returns the resulting real-world dimensions for the scene.json entry."""

    bounds = scene.bounds  # shape (2, 3): [min_xyz, max_xyz]
    size = bounds[1] - bounds[0]

    if np.any(size <= 0):
        raise ValueError(f"Mesh has invalid bounds: size={size}")

    # 1. Scale first (so subsequent centering math is in final units)
    scale_factor = target_width / size[0]
    scene.apply_transform(trimesh.transformations.scale_matrix(scale_factor))

    # 2. Recompute bounds after scaling, recenter x/z, floor-align y
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    translation = [-center[0], -bounds[0][1], -center[2]]
    scene.apply_transform(trimesh.transformations.translation_matrix(translation))

    final_bounds = scene.bounds
    final_size = final_bounds[1] - final_bounds[0]

    return {
        "width": round(float(final_size[0]), 4),
        "height": round(float(final_size[1]), 4),
        "depth": round(float(final_size[2]), 4),
    }


def convert(obj_path: Path, out_path: Path, target_width: float) -> dict:
    scene = load_as_scene(obj_path)
    dimensions = normalize(scene, target_width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path, file_type="glb")

    return dimensions


def build_scene_entry(
    obj_id: str,
    model_rel_path: str,
    dimensions: dict,
    position: dict,
    rotation: dict | None,
) -> dict:
    entry = {
        "id": obj_id,
        "type": "model",
        "model": model_rel_path,
        "position": position,
        "dimensions": dimensions,
    }
    if rotation:
        entry["rotation"] = rotation
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, type=Path, help="Path to TripoSR mesh.obj")
    parser.add_argument("--out", required=True, type=Path, help="Output .glb path")
    parser.add_argument("--target-width", required=True, type=float, help="Real-world width (x-axis) in scene units/meters")
    parser.add_argument("--id", default=None, help="Object id for scene.json entry")
    parser.add_argument("--model-rel-path", default=None, help="Path to write into scene.json's 'model' field (defaults to --out)")
    parser.add_argument("--pos-x", type=float, default=0.0)
    parser.add_argument("--pos-y", type=float, default=0.0)
    parser.add_argument("--pos-z", type=float, default=0.0)
    parser.add_argument("--rot-y", type=float, default=None, help="Y rotation in degrees, omitted if not given")
    parser.add_argument("--scene-json", type=Path, default=None, help="If given, appends/updates the entry in this scene.json")
    args = parser.parse_args()

    if not args.obj.exists():
        print(f"ERROR: {args.obj} does not exist", file=sys.stderr)
        sys.exit(1)

    dimensions = convert(args.obj, args.out, args.target_width)

    obj_id = args.id or args.out.stem
    model_rel_path = args.model_rel_path or str(args.out)
    rotation = {"x": 0, "y": args.rot_y, "z": 0} if args.rot_y is not None else None

    entry = build_scene_entry(
        obj_id=obj_id,
        model_rel_path=model_rel_path,
        dimensions=dimensions,
        position={"x": args.pos_x, "y": args.pos_y, "z": args.pos_z},
        rotation=rotation,
    )

    print(json.dumps(entry, indent=2))

    if args.scene_json:
        if args.scene_json.exists():
            data = json.loads(args.scene_json.read_text())
        else:
            data = {"objects": []}

        data.setdefault("objects", [])
        # replace existing entry with same id, else append
        data["objects"] = [o for o in data["objects"] if o.get("id") != obj_id]
        data["objects"].append(entry)

        args.scene_json.write_text(json.dumps(data, indent=2))
        print(f"\nUpdated {args.scene_json}")


if __name__ == "__main__":
    main()