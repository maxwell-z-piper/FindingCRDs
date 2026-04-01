import os
import requests
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
output_dir = Path(os.path.join(os.getcwd(), 'CUBES'))
output_dir.mkdir(parents=True, exist_ok=True)

BASE        = "https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/"
DRPALL_URL  = BASE + "drpall-v3_1_1.fits"
FILE_TYPES  = ["LOGCUBE", "LOGRSS"]
MAX_WORKERS = 6  

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_all_plateifus():
    drpall_path = Path(os.getcwd()) / "drpall-v3_1_1.fits"
    if drpall_path.exists():
        print(f"DRPall already present, using cached copy.")
    else:
        print(f"Downloading DRPall from {DRPALL_URL} ...")
        r = requests.get(DRPALL_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(drpall_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"  ✓ Saved ({drpall_path.stat().st_size / 1e6:.1f} MB)\n")

    with fits.open(drpall_path) as hdu:
        plateifus = [p.strip() for p in hdu[1].data["PLATEIFU"].tolist()]
    print(f"Found {len(plateifus)} plate-ifu pairs.\n")
    return plateifus

def download_one(task):
    """Download a single (plateifu, filetype) pair. Returns (task_id, error|None)."""
    plateifu, filetype = task
    plate, ifu = plateifu.split("-")
    url      = f"{BASE}{plate}/stack/manga-{plateifu}-{filetype}.fits.gz"
    out_path = output_dir / f"manga-{plateifu}-{filetype}.fits.gz"

    if out_path.exists():
        return (f"{plateifu}-{filetype}", None)  # already done

    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return (f"{plateifu}-{filetype}", None)
    except requests.exceptions.RequestException as e:
        return (f"{plateifu}-{filetype}", str(e))

def main():
    plateifus = get_all_plateifus()
    tasks  = [(p, ft) for p in plateifus for ft in FILE_TYPES]
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                task_id, error = future.result()
                if error:
                    failed.append(task_id)
                pbar.update(1)

    print(f"\nDone. {len(tasks) - len(failed)}/{len(tasks)} files downloaded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for p in failed:
            print(f"  {p}")
        import numpy as np
        np.savetxt(output_dir / "failed_downloads.txt", failed, fmt="%s")

if __name__ == "__main__":
    main()
