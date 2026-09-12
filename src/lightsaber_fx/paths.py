import shutil
from pathlib import Path

import platformdirs

APP_NAME = "lightsaber-fx"


def get_data_dir() -> Path:
    d = Path(platformdirs.user_data_dir(APP_NAME))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sam2_src_dir() -> Path:
    return get_data_dir() / "sam2-src"


def get_checkpoint_path() -> Path:
    return get_data_dir() / "checkpoints" / "sam2.1_hiera_small.pt"


def get_jobs_dir() -> Path:
    d = get_data_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_job_dir(job_id: str) -> Path:
    d = get_jobs_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_jobs() -> int:
    count = 0
    for child in get_jobs_dir().iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            count += 1
    return count
