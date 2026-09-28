"""
Vinyasa Kaggle TripoSR Automation

Project structure:

reconstruction/
├── input/
├── kaggle_kernel/
├── output/
└── kaggle_automation.py

Existing Kaggle resources:
    Dataset: shrishlok/vinyasa-object-crops
    Kernel:  shrishlok/vinyasa-triposr-object-gen

Batch version: kernel now processes every object image found in the
input dataset (plus an optional object_manifest.json for placement),
so this driver retrieves ALL generated GLBs/scene entries, keyed by
object id, instead of assuming a single "sofa.png" -> one GLB.
"""

from __future__ import annotations
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if sys.stderr is not None:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "input"
KERNEL_DIR = PROJECT_DIR / "kaggle_kernel"
OUTPUT_DIR = PROJECT_DIR / "output"

DATASET_REF = "shrishlok/vinyasa-object-crops"
KERNEL_REF = "shrishlok/vinyasa-triposr-object-gen"

# Polling configuration.
POLL_INTERVAL = 15
TIMEOUT_SECONDS = 1800


# ============================================================
# KAGGLE CLI HELPER
# ============================================================

def run_kaggle(command: list[str], cwd: Path) -> str:
    """
    Execute a Kaggle CLI command.

    Args:
        command: Full command as a list.
        cwd: Directory from which to execute it.

    Returns:
        stdout as a string.

    Raises:
        RuntimeError: If Kaggle CLI fails.
    """

    print("\n" + "=" * 60)
    print("RUNNING COMMAND")
    print("=" * 60)

    print("Directory:", cwd)
    print("Command:", " ".join(command))
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle command failed with exit code "
            f"{result.returncode}:\n"
            f"{' '.join(command)}"
        )

    return result.stdout


# ============================================================
# 1. PUSH INPUT DATASET
# ============================================================

def push_input_dataset() -> str:
    """
    Create the Kaggle dataset for the first time.

    Existing command:
        kaggle datasets create -p .

    Executed from:
        reconstruction/input/
    """

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    metadata = INPUT_DIR / "dataset-metadata.json"

    if not metadata.exists():
        raise FileNotFoundError(
            f"Missing dataset-metadata.json: {metadata}"
        )

    return run_kaggle(
        [
            "kaggle",
            "datasets",
            "create",
            "-p",
            ".",
        ],
        cwd=INPUT_DIR,
    )


# ============================================================
# 2. UPDATE EXISTING DATASET
# ============================================================

def update_input_dataset(
    message: str = "Update Vinyasa object crops",
) -> str:
    """
    Create a new version of the existing Kaggle dataset.

    Use this when the dataset already exists.

    Command:
        kaggle datasets version -p . -m "message"

    Executed from:
        reconstruction/input/
    """

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    return run_kaggle(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            ".",
            "-m",
            message,
        ],
        cwd=INPUT_DIR,
    )


# ============================================================
# 3. LIST DATASET FILES
# ============================================================

def list_dataset_files() -> str:
    """
    Existing command:
        kaggle datasets files shrishlok/vinyasa-object-crops
    """

    return run_kaggle(
        [
            "kaggle",
            "datasets",
            "files",
            DATASET_REF,
        ],
        cwd=PROJECT_DIR,
    )


# ============================================================
# 4. PUSH / START TRIPOSR KERNEL
# ============================================================

def start_triposr() -> str:
    """
    Push the Kaggle kernel.

    Existing command:
        kaggle kernels push -p .

    Executed from:
        reconstruction/kaggle_kernel/
    """

    if not KERNEL_DIR.exists():
        raise FileNotFoundError(
            f"Kernel directory not found: {KERNEL_DIR}"
        )

    metadata = KERNEL_DIR / "kernel-metadata.json"

    if not metadata.exists():
        raise FileNotFoundError(
            f"Missing kernel-metadata.json: {metadata}"
        )

    return run_kaggle(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            ".",
        ],
        cwd=KERNEL_DIR,
    )


# ============================================================
# 5. CHECK KERNEL STATUS
# ============================================================

def check_status() -> str:
    """
    Check the status of the Kaggle kernel.

    Command:
        kaggle kernels status
        shrishlok/vinyasa-triposr-object-gen
    """

    return run_kaggle(
        [
            "kaggle",
            "kernels",
            "status",
            KERNEL_REF,
        ],
        cwd=PROJECT_DIR,
    )


# ============================================================
# 6. WAIT FOR COMPLETION
# ============================================================

def wait_for_completion() -> str:
    """
    Poll the Kaggle kernel until it finishes.

    Returns:
        Final status output.

    Raises:
        TimeoutError: If execution exceeds timeout.
        RuntimeError: If the kernel fails.
    """

    started_at = time.monotonic()

    while True:

        elapsed = time.monotonic() - started_at

        if elapsed > TIMEOUT_SECONDS:
            raise TimeoutError(
                "TripoSR Kaggle kernel timed out after "
                f"{TIMEOUT_SECONDS} seconds."
            )

        status_output = check_status()

        status = status_output.lower()

        print("\nCurrent Kaggle status:")
        print(status_output)

        # Kaggle commonly reports "complete" on success.
        if "complete" in status:
            print("\nTripoSR completed successfully.")
            return status_output

        # Explicit failure indicators.
        if any(
            word in status
            for word in [
                "error",
                "failed",
                "cancelled",
            ]
        ):
            raise RuntimeError(
                "TripoSR Kaggle kernel failed:\n"
                f"{status_output}"
            )

        print(
            f"\nStill running. "
            f"Checking again in {POLL_INTERVAL} seconds..."
        )

        time.sleep(POLL_INTERVAL)


# ============================================================
# 7. RETRIEVE KAGGLE OUTPUT
# ============================================================

def retrieve_output() -> Path:
    """
    Download Kaggle output and delete unnecessary files.

    Keeps:
        .glb
        scene_entry_*.json

    Deletes:
        .png
        .jpg
        .jpeg
        .mp4
        .obj
        .mtl
        .webp
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_kaggle(
        [
            "kaggle",
            "kernels",
            "output",
            KERNEL_REF,
            "-p",
            str(OUTPUT_DIR),
        ],
        cwd=PROJECT_DIR,
    )

    print("\nCleaning unnecessary files...")

    removable_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".mp4",
        ".obj",
        ".mtl",
        ".webp",
    }

    deleted_count = 0

    for file in OUTPUT_DIR.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() in removable_extensions:

            print(f"[DELETE] {file}")

            file.unlink()

            deleted_count += 1

    # Remove empty directories left behind.
    for directory in sorted(
        OUTPUT_DIR.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    ):

        if directory.is_dir():

            try:
                directory.rmdir()
            except OSError:
                pass

    print(
        f"Cleanup complete. Deleted {deleted_count} files."
    )

    return OUTPUT_DIR


# ============================================================
# 8. FIND GENERATED GLBs / SCENE ENTRIES (batch-aware)
# ============================================================

def find_generated_glbs() -> dict[str, Path]:
    """
    Find all generated GLBs in output/, keyed by object id.

    tripo_kernel.py now writes one GLB per source image, named
    after the image's stem (e.g. sofa.glb, chair.glb).
    """

    if not OUTPUT_DIR.exists():
        return {}

    return {
        glb_path.stem: glb_path
        for glb_path in OUTPUT_DIR.rglob("*.glb")
    }


def find_scene_entries() -> dict[str, Path]:
    """
    Find all scene_entry_*.json files in output/, keyed by object id.
    """

    if not OUTPUT_DIR.exists():
        return {}

    entries: dict[str, Path] = {}

    for json_path in OUTPUT_DIR.rglob("scene_entry_*.json"):
        object_id = json_path.stem.removeprefix("scene_entry_")
        entries[object_id] = json_path

    return entries


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

def generate_3d_models(
    update_dataset: bool = False,
    dataset_message: str = "Update Vinyasa object crops",
) -> dict:
    """
    Execute the full Kaggle TripoSR batch pipeline.

    Current workflow:
        1. Optionally update dataset (push new crops + manifest)
        2. Push kernel
        3. Wait for completion
        4. Retrieve output
        5. Find all GLBs + scene entries, keyed by object id

    Returns:
        {
          "glbs": {object_id: Path, ...},
          "scene_entries": {object_id: Path, ...},
        }
    """

    print("\n" + "=" * 60)
    print("VINYASA TRIPOSR BATCH PIPELINE STARTED")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: DATASET
    # --------------------------------------------------------

    if update_dataset:
        print("\n[1/5] Updating Kaggle input dataset...")

        update_input_dataset(
            message=dataset_message,
        )

    else:
        print(
            "\n[1/5] Using existing Kaggle dataset."
        )

    # --------------------------------------------------------
    # STEP 2: START KERNEL
    # --------------------------------------------------------

    print("\n[2/5] Starting TripoSR kernel...")

    start_triposr()

    # --------------------------------------------------------
    # STEP 3: WAIT
    # --------------------------------------------------------

    print("\n[3/5] Waiting for Kaggle completion...")

    wait_for_completion()

    # --------------------------------------------------------
    # STEP 4: RETRIEVE
    # --------------------------------------------------------

    print("\n[4/5] Retrieving generated output...")

    retrieve_output()

    # --------------------------------------------------------
    # STEP 5: FIND GLBs + SCENE ENTRIES
    # --------------------------------------------------------

    print("\n[5/5] Searching for generated GLBs and scene entries...")

    glb_paths = find_generated_glbs()
    scene_entries = find_scene_entries()

    if not glb_paths:
        raise FileNotFoundError(
            "Kaggle completed, but no .glb files were found "
            f"in {OUTPUT_DIR}."
        )

    missing_entries = set(glb_paths) - set(scene_entries)
    if missing_entries:
        print(
            f"[WARNING] {len(missing_entries)} object(s) have a GLB "
            f"but no scene_entry JSON: {sorted(missing_entries)}"
        )

    print("\n" + "=" * 60)
    print("VINYASA TRIPOSR BATCH PIPELINE COMPLETE")
    print("=" * 60)

    print(f"Generated {len(glb_paths)} object(s):")
    for object_id, glb_path in glb_paths.items():
        print(f"  - {object_id}: {glb_path}")

    return {
        "glbs": glb_paths,
        "scene_entries": scene_entries,
    }


# ============================================================
# CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    # Manifest test run:
    # input/ now has sofa.png, chair.jpg, and object_manifest.json.
    # update_dataset=True so Kaggle actually receives the new files
    # before the kernel runs — without this, it'll reuse the last
    # pushed dataset version and never see chair.jpg or the manifest.
    result = generate_3d_models(
        update_dataset=True,
        dataset_message="Add chair crop + object_manifest.json for batch test",
    )

    print("\nFINAL RESULTS:")
    print("GLBs:", result["glbs"])
    print("Scene entries:", result["scene_entries"])

    import importlib.util

    merge_scene_path = PROJECT_DIR / "script" / "merge_scene.py"
    spec = importlib.util.spec_from_file_location("merge_scene", merge_scene_path)
    merge_scene_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge_scene_module)

    print("\nMerging scene entries into rendering/src/data/scene.json...")
    merge_scene_module.merge_scene()