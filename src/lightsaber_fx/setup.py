import shutil
import subprocess
import sys

from .paths import get_checkpoint_path, get_sam2_src_dir

SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"


def bootstrap(force=False):
    sam2_src = get_sam2_src_dir()
    if force and sam2_src.exists():
        shutil.rmtree(sam2_src)

    if not sam2_src.exists():
        print(f"Cloning SAM2 into {sam2_src} ...")
        subprocess.run(["git", "clone", "-q", SAM2_REPO_URL, str(sam2_src)], check=True)
    else:
        print(f"SAM2 already cloned at {sam2_src}, skipping clone.")

    print("Installing SAM2 (pip install -e .) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(sam2_src)], check=True)

    checkpoint_path = get_checkpoint_path()
    if checkpoint_path.exists() and not force:
        print(f"Checkpoint already present at {checkpoint_path}, skipping download.")
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading small SAM2.1 checkpoint to {checkpoint_path} ...")
    subprocess.run(["curl", "-fL", "-o", str(checkpoint_path), CHECKPOINT_URL], check=True)
