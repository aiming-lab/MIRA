"""
Download the full MIRA dataset from Hugging Face to ./MIRA.
Uses per-task downloads with 60s delay between tasks to stay under HF rate limits.
"""

import time

from huggingface_hub import snapshot_download

REPO_ID = "YiyangAiLab/MIRA"
LOCAL_DIR = "./MIRA"
DELAY_BETWEEN_TASKS = 60
MAX_WORKERS = 1

TASKS = [
    "billiards",
    "convex_hull",
    "cubes_count",
    "cubes_missing",
    "defuse_a_bomb",
    "electric_charge",
    "gear_rotation",
    "localizer",
    "mirror_clock",
    "mirror_pattern",
    "multi_piece_puzzle",
    "overlap",
    "paper_airplane",
    "puzzle",
    "rolling_dice_sum",
    "rolling_dice_top",
    "rolling_dice_two",
    "trailer_cubes_count",
    "trailer_cubes_missing",
    "unfolded_cube",
]


def main():
    print(f"Downloading {len(TASKS)} tasks to {LOCAL_DIR} ({DELAY_BETWEEN_TASKS}s between tasks).")
    for i, task in enumerate(TASKS):
        print(f"  [{i+1}/{len(TASKS)}] {task} ...")
        patterns = [f"{task}/*", f"{task}/*/*", f"{task}/*/*/*"]
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            allow_patterns=patterns,
            max_workers=MAX_WORKERS,
        )
        if i < len(TASKS) - 1:
            time.sleep(DELAY_BETWEEN_TASKS)
    print(f"Done. MIRA dataset in {LOCAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
