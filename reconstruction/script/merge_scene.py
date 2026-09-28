"""
merge_scene.py — combines all scene_entry_*.json from a Part 2 batch
run into the single scene.json Part 3 reads, and copies GLBs into
rendering/public/model/.

Run this after kaggle_automation.py's generate_3d_models() completes.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "reconstruction" / "output"
RENDERING_MODEL_DIR = PROJECT_ROOT / "rendering" / "public" / "model"
RENDERING_SCENE_JSON = PROJECT_ROOT / "rendering" / "src" / "data" / "scene.json"


def model_url(object_id: str) -> str:
    # Vite serves files in rendering/public/ from the site root.
    return f"/model/{object_id}.glb"


def merge_scene() -> Path:
    RENDERING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RENDERING_SCENE_JSON.parent.mkdir(parents=True, exist_ok=True)

    objects = []

    for entry_path in sorted(OUTPUT_DIR.rglob("scene_entry_*.json")):
        entry = json.loads(entry_path.read_text())
        object_id = entry["id"]

        glb_source = OUTPUT_DIR / f"{object_id}.glb"
        if not glb_source.exists():
            print(
                f"[WARNING] {object_id}: scene_entry exists but "
                f"{glb_source.name} is missing — skipping"
            )
            continue

        glb_dest = RENDERING_MODEL_DIR / f"{object_id}.glb"
        shutil.copy2(glb_source, glb_dest)

        entry["model"] = model_url(object_id)
        objects.append(entry)
        print(f"[MERGED] {object_id}")

    scene = {"objects": objects}
    RENDERING_SCENE_JSON.write_text(json.dumps(scene, indent=2) + "\n")

    print(f"\nWrote {len(objects)} object(s) to {RENDERING_SCENE_JSON}")
    return RENDERING_SCENE_JSON


if __name__ == "__main__":
    merge_scene()
