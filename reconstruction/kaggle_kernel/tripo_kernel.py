"""
tripo_kernel.py — runs on Kaggle (GPU kernel).

Pipeline:
  1. Validate input image immediately before downloading or compiling anything.
  2. Set up TripoSR + patched torchmcubes (fast minimal dependencies).
  3. Run TripoSR on the object crop image.
  4. Convert mesh.obj -> normalized .glb (scale to target width, floor-align, recenter).
  5. Write the .glb + scene.json entry snippet to /kaggle/working.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# ==============================================================================
# CONFIG — edit these per object/run
# ==============================================================================
INPUT_IMAGE = "/kaggle/input/vinyasa-object-crops/sofa.png"
OBJECT_ID = "sofa"
TARGET_WIDTH = 2.8                          # real-world width (x-axis) in Three.js scene units
POSITION = {"x": 0, "y": 0, "z": -0.6}
ROTATION_Y = 180                            # degrees, or None to omit rotation
MODEL_REL_PATH = f"public/model/{OBJECT_ID}.glb"

WORK_DIR = Path("/kaggle/working")
TRIPOSR_DIR = WORK_DIR / "TripoSR"
MCUBES_DIR = WORK_DIR / "torchmcubes"


def run(cmd: str):
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def resolve_input_image(target_path: str, object_id: str) -> Path:
    """Check if the image exists. If not, scan /kaggle/input and auto-discover."""
    path = Path(target_path)
    if path.is_file():
        return path

    print(f"\n[WARNING] Specified path '{target_path}' not found. Scanning /kaggle/input ...")
    input_dir = Path("/kaggle/input")
    all_files = list(input_dir.rglob("*")) if input_dir.exists() else []

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
    print("====================================")

    # Attempt to auto-locate matching object image
    candidates = [
        f for f in file_paths
        if f.stem.lower() == object_id.lower() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]
    if candidates:
        selected = candidates[0]
        print(f"[AUTO-RESOLVED] Using detected file: {selected}\n")
        return selected

    raise FileNotFoundError(
        f"Could not find an image for '{object_id}' in /kaggle/input. "
        f"Available files listed above."
    )


def setup_triposr():
    if not TRIPOSR_DIR.exists():
        run(f"git clone https://github.com/VAST-AI-Research/TripoSR.git {TRIPOSR_DIR}")

    # Build tools for torchmcubes
    run("pip install -q scikit-build-core cmake ninja pybind11")

    # Minimal headless dependencies without pulling in heavy web UIs (gradio)
    # or downgrading numpy/transformers
    run("pip install -q trimesh omegaconf einops rembg pygltflib xatlas")

    # Patch pymatting import conflict if needed
    pymatting_init = Path("/usr/local/lib/python3.12/dist-packages/pymatting/foreground/__init__.py")
    if pymatting_init.exists():
        run(f"sed -i '/estimate_foreground_ml_cupy/d' {pymatting_init}")

    # Build and install patched torchmcubes
    if MCUBES_DIR.exists():
        run(f"rm -rf {MCUBES_DIR}")
    run(f"git clone https://github.com/tatsy/torchmcubes.git {MCUBES_DIR}")
    run(f"sed -i 's/inline __device__ __host__ float lerp(/inline __device__ __host__ float lerp_custom(/' {MCUBES_DIR}/cxx/helper_math.h")
    run(f"pip install -q --no-build-isolation --no-cache-dir {MCUBES_DIR}")


def run_triposr(image_path: Path) -> Path:
    out_dir = WORK_DIR / "tripo_output"
    run(
        f'python {TRIPOSR_DIR}/run.py "{image_path}" '
        f'--output-dir {out_dir}/ --bake-texture --render'
    )
    mesh_path = out_dir / "0" / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Expected mesh at {mesh_path} — TripoSR run may have failed")
    return mesh_path


def load_as_scene(obj_path: Path):
    import trimesh
    loaded = trimesh.load(obj_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    return loaded


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


def main():
    # 1. FAST FAIL: Validate image path before running any heavy setups
    image_path = resolve_input_image(INPUT_IMAGE, OBJECT_ID)

    # 2. Setup dependencies and model repo
    setup_triposr()

    # 3. Inference
    mesh_path = run_triposr(image_path)

    # 4. Post-processing to GLB
    glb_path = WORK_DIR / f"{OBJECT_ID}.glb"
    dimensions = convert_to_glb(mesh_path, glb_path, TARGET_WIDTH)

    # 5. Scene metadata JSON
    entry = build_scene_entry(OBJECT_ID, MODEL_REL_PATH, dimensions, POSITION, ROTATION_Y)
    entry_path = WORK_DIR / f"scene_entry_{OBJECT_ID}.json"
    entry_path.write_text(json.dumps(entry, indent=2))

    print("\n=== DONE ===")
    print(f"GLB written to: {glb_path}")
    print(f"Scene entry written to: {entry_path}")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()