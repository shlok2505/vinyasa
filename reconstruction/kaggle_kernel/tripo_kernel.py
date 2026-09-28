"""
tripo_kernel.py — runs on Kaggle (GPU kernel).

Batch pipeline:
  1. Discover all object crop images in /kaggle/input (one dataset version
     can now hold multiple objects, not just "sofa.png").
  2. Set up TripoSR + patched torchmcubes ONCE for the whole kernel run.
  3. For each object image:
       a. Run TripoSR reconstruction.
       b. Convert mesh.obj -> normalized .glb (scale, floor-align, recenter).
       c. Write <object_id>.glb + scene_entry_<object_id>.json to /kaggle/working.
     A failure on one object is logged and skipped — it does not abort
     the rest of the batch.
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path


# ============================================================
# CONFIG — edit per project/dataset layout
# ============================================================
INPUT_DIR = Path("/kaggle/input/vinyasa-object-crops")
WORK_DIR = Path("/kaggle/working")
TEMP_DIR = Path("/tmp")
TRIPOSR_DIR = TEMP_DIR / "TripoSR"
MCUBES_DIR = TEMP_DIR / "torchmcubes"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Optional manifest produced by Part 1: maps object_id -> per-object
# placement metadata. If absent, or an object_id is missing from it,
# the DEFAULT_* values below are used instead, so a batch run never
# fails just because placement data isn't ready yet.
MANIFEST_FILENAME = "object_manifest.json"

DEFAULT_TARGET_WIDTH = 1.2
DEFAULT_POSITION = {"x": 0, "y": 0, "z": 0}
DEFAULT_ROTATION_Y = 0
MODEL_REL_PATH_TEMPLATE = "/model/{object_id}.glb"


def run(cmd: str):
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def discover_input_images() -> list[Path]:
    """
    Find every object crop image under /kaggle/input.

    Replaces the old single-file resolve_input_image(): instead of
    looking for one hardcoded filename, this scans the whole mounted
    dataset and processes every image found, so Part 1 can push any
    number of cropped objects in one dataset version.
    """
    search_dir = INPUT_DIR if INPUT_DIR.exists() else Path("/kaggle/input")
    all_files = list(search_dir.rglob("*")) if search_dir.exists() else []
    file_paths = [p for p in all_files if p.is_file()]

    if not file_paths:
        raise FileNotFoundError(
            "\n[FATAL] /kaggle/input is completely empty!\n"
            "Your dataset is not mounted. Check 'dataset_sources' in kernel-metadata.json:\n"
            '  "dataset_sources": ["<username>/vinyasa-object-crops"]\n'
        )

    print("=== Found files in /kaggle/input ===")
    for f in file_paths:
        print(f"  {f}")
    print("=" * 40)

    images = sorted(
        p for p in file_paths if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No object crop images (extensions {IMAGE_EXTENSIONS}) found "
            f"under {search_dir}. Files present are listed above."
        )

    print(f"\n[DISCOVERY] {len(images)} object image(s) to process:")
    for img in images:
        print(f"  - {img.stem}  ({img})")

    return images


def load_manifest() -> dict:
    """
    Load the optional per-object placement manifest, if Part 1 provided one.

    Expected shape:
        {
          "sofa":     {"target_width": 1.2, "position": {"x":0,"y":0,"z":-1}, "rotation_y": 90},
          "chair_01": {"target_width": 0.5, "position": {"x":1,"y":0,"z":0.5}, "rotation_y": 0}
        }

    A missing file, or a missing object_id within it, falls back to
    DEFAULT_* above.
    """
    search_dir = INPUT_DIR if INPUT_DIR.exists() else Path("/kaggle/input")
    candidates = list(search_dir.rglob(MANIFEST_FILENAME)) if search_dir.exists() else []

    if not candidates:
        print(f"[MANIFEST] No {MANIFEST_FILENAME} found — using defaults for all objects.")
        return {}

    manifest_path = candidates[0]
    try:
        manifest = json.loads(manifest_path.read_text())
        print(f"[MANIFEST] Loaded {manifest_path} with {len(manifest)} object entr(y/ies).")
        return manifest
    except Exception as e:
        print(f"[MANIFEST] Failed to parse {manifest_path}: {e} — using defaults for all objects.")
        return {}


def patch_bake_texture_device_bug():
    """Fix an unmerged upstream bug: bake_texture.py's positions_to_colors
    builds the 'positions' tensor with no device (defaults to CPU), so on a
    GPU kernel it stays on CPU while query_triplane's grid_sample runs on
    CUDA, raising a device-mismatch RuntimeError. Also patches the resulting
    queried_grid["color"].numpy() call, which would otherwise fail once that
    tensor is on CUDA (numpy() requires a CPU tensor).
    """
    bake_path = TRIPOSR_DIR / "tsr" / "bake_texture.py"
    if not bake_path.exists():
        print(f"[PATCH] {bake_path} not found, skipping")
        return

    text = bake_path.read_text()
    if "device=scene_code.device" in text:
        print("[PATCH] bake_texture.py already patched, skipping")
        return

    old_positions_line = "positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1])"
    new_positions_line = (
        "positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1], "
        "dtype=torch.float32, device=scene_code.device)"
    )
    old_color_line = 'rgb_f = queried_grid["color"].numpy().reshape(-1, 3)'
    new_color_line = 'rgb_f = queried_grid["color"].cpu().numpy().reshape(-1, 3)'

    if old_positions_line not in text:
        print("[PATCH] bake_texture.py: expected positions line not found, skipping "
              "(upstream code may have changed — check manually if the device error recurs)")
        return

    text = text.replace(old_positions_line, new_positions_line, 1)

    if old_color_line in text:
        text = text.replace(old_color_line, new_color_line, 1)
    else:
        print("[PATCH] bake_texture.py: color .numpy() line not found in expected "
              "form — positions patched, but check for a CUDA .numpy() crash downstream")

    bake_path.write_text(text)
    print("[PATCH] bake_texture.py: pinned positions tensor to scene_code.device "
          "and moved color tensor to CPU before .numpy()")


def setup_triposr():
    if not TRIPOSR_DIR.exists():
        run(f"git clone https://github.com/VAST-AI-Research/TripoSR.git {TRIPOSR_DIR}")
        run(f"cat {TRIPOSR_DIR}/run.py")

    # xvfb gives moderngl a virtual X display so bake_texture's
    # moderngl.create_context(standalone=True) doesn't fail with
    # "XOpenDisplay: cannot open display" on Kaggle's headless kernel
    run("apt-get update -y -q")
    run("apt-get install -y -q xvfb")

    # Build tools for torchmcubes
    run("pip install -q scikit-build-core cmake ninja pybind11")

    # Minimal headless dependencies with onnxruntime for rembg and moderngl for texture baking
    run('pip install -q trimesh omegaconf einops "rembg[cpu]" onnxruntime pygltflib xatlas moderngl')

    # Patch pymatting import conflict if needed
    run("python -c \"import pymatting, pathlib; p = pathlib.Path(pymatting.__file__).parent / 'foreground' / '__init__.py'; [p.write_text('\\n'.join(l for l in p.read_text().splitlines() if 'estimate_foreground_ml_cupy' not in l) + '\\n') if p.exists() else None]\"")

    # Patch known TripoSR bug: bake_texture builds the positions tensor without
    # pinning it to scene_code's device, so on GPU it stays on CPU while
    # query_triplane's grid_sample runs on CUDA -> "different device" RuntimeError.
    # Unmerged upstream fix: https://github.com/VAST-AI-Research/TripoSR/pull/137
    patch_bake_texture_device_bug()

    # Build and install patched torchmcubes
    if MCUBES_DIR.exists():
        run(f"rm -rf {MCUBES_DIR}")
    run(f"git clone https://github.com/tatsy/torchmcubes.git {MCUBES_DIR}")
    run(f"sed -i 's/inline __device__ __host__ float lerp(/inline __device__ __host__ float lerp_custom(/' {MCUBES_DIR}/cxx/helper_math.h")
    run(f"pip install -q --no-build-isolation --no-cache-dir {MCUBES_DIR}")


def run_triposr(image_path: Path, object_id: str) -> Path:
    # Namespaced by object_id so concurrent objects in the same batch
    # never collide on TripoSR's own "0/mesh.obj" output convention.
    out_dir = WORK_DIR / "tripo_output" / object_id
    run(
        f'xvfb-run -a python {TRIPOSR_DIR}/run.py "{image_path}" '
        f'--output-dir {out_dir}/ --bake-texture --render '
        f'--mc-resolution 320'
    )
    mesh_path = out_dir / "0" / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Expected mesh at {mesh_path} — TripoSR run may have failed")
    return mesh_path


def load_as_scene(obj_path: Path):
    """Load TripoSR's OBJ (which has UVs but no .mtl) and manually
    attach the adjacent texture.png using those UVs.

    Uses an explicit non-metallic PBR material: glTF defaults metallicFactor
    to 1.0 when unset, which renders dark without an env map and washed out
    in viewers that have one."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from PIL import Image
    mesh = trimesh.load(obj_path, process=False, force="mesh")

    texture_path = obj_path.parent / "texture.png"
    uv = getattr(mesh.visual, "uv", None)

    if texture_path.exists() and uv is not None:
        image = Image.open(texture_path)
        material = PBRMaterial(
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        print(f"[TEXTURE] Attached {texture_path.name} using {len(uv)} UVs (metallic=0, roughness=1)")
    else:
        print(f"[WARN] No texture.png or no UVs found next to {obj_path} — exporting untextured")

    return trimesh.Scene(mesh)


def normalize(scene, target_width: float) -> dict:
    import numpy as np
    import trimesh

    bounds = scene.bounds
    size = bounds[1] - bounds[0]
    if np.any(size <= 0):
        raise ValueError(f"Mesh has invalid bounds: size={size}")

    scale_factor = target_width / size[0]
    scene.apply_transform(trimesh.transformations.scale_matrix(scale_factor))

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


def convert_to_glb(obj_path: Path, out_path: Path, target_width: float) -> dict:
    scene = load_as_scene(obj_path)
    dimensions = normalize(scene, target_width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path, file_type="glb")
    return dimensions


def build_scene_entry(obj_id, model_rel_path, dimensions, position, rotation_y):
    entry = {
        "id": obj_id,
        "type": "model",
        "model": model_rel_path,
        "position": position,
        "dimensions": dimensions,
    }
    if rotation_y is not None:
        entry["rotation"] = {"x": 0, "y": rotation_y, "z": 0}
    return entry


def process_object(image_path: Path, manifest: dict) -> dict:
    """Runs the full single-object pipeline (old steps 3-5) for one image.
    Raises on failure — main() decides whether to skip and continue."""
    object_id = image_path.stem
    object_config = manifest.get(object_id, {})

    target_width = object_config.get("target_width", DEFAULT_TARGET_WIDTH)
    position = object_config.get("position", DEFAULT_POSITION)
    rotation_y = object_config.get("rotation_y", DEFAULT_ROTATION_Y)
    model_rel_path = MODEL_REL_PATH_TEMPLATE.format(object_id=object_id)

    print(f"\n--- Processing '{object_id}' ({image_path.name}) ---")
    print(f"    target_width={target_width} position={position} rotation_y={rotation_y}")

    mesh_path = run_triposr(image_path, object_id)
    
    obj_text = mesh_path.read_text()
    prefix_counts = {}
    for line in obj_text.splitlines():
        prefix = line.split()[0] if line.split() else ""
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    print(f"[DEBUG] mesh.obj line prefix counts: {prefix_counts}")
    print(f"[DEBUG] total lines: {sum(prefix_counts.values())}")        

    glb_path = WORK_DIR / f"{object_id}.glb"
    dimensions = convert_to_glb(mesh_path, glb_path, target_width)

    entry = build_scene_entry(object_id, model_rel_path, dimensions, position, rotation_y)
    entry_path = WORK_DIR / f"scene_entry_{object_id}.json"
    entry_path.write_text(json.dumps(entry, indent=2))

    print(f"GLB written to: {glb_path}")
    print(f"Scene entry written to: {entry_path}")
    print(json.dumps(entry, indent=2))

    return entry


def main():
    # 1. FAST FAIL: discover every object image before running any heavy setup
    images = discover_input_images()
    manifest = load_manifest()

    # 2. Setup dependencies and model repo — ONCE for the whole batch
    setup_triposr()

    # 3-5. Per-object: inference -> GLB -> scene entry.
    # A failure on one object is logged and skipped, not fatal to the batch.
    succeeded, failed = [], []
    for image_path in images:
        object_id = image_path.stem
        try:
            process_object(image_path, manifest)
            succeeded.append(object_id)
        except Exception as e:
            print(f"[FAILED] {object_id}: {e}", flush=True)
            failed.append(object_id)
            continue

    print("\n=== BATCH DONE ===")
    print(f"Succeeded ({len(succeeded)}): {succeeded}")
    print(f"Failed    ({len(failed)}): {failed}")

    if not succeeded:
        raise RuntimeError("All objects in the batch failed — see logs above.")


if __name__ == "__main__":
    main()