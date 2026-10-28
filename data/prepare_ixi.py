# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "hd-bet",
#   "torch",
#   "nibabel",
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "huggingface_hub>=0.30",
# ]
# ///
"""
Phase 1 data preparation for the severity-robust-mocorr project.

IXI T1 volumes -> skull-stripped (HD-BET), 1 mm isotropic, RAS-oriented,
intensity-normalised volumes with brain masks, plus a subject-level,
site-stratified train/val/test split.

Outputs (pushed to a private Hub dataset repo):
  volumes/<IXI_ID>.npz   image (float16, [0,1]) and mask (uint8); axial slices = [:, :, k]
  splits.json            {"train": [...], "val": [...], "test": [...]}
  manifest.csv           id, site, shape, split
  qc_montage.png         mid-axial slices with brain-mask outline
  README.md              provenance and licence note
"""
import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, label, zoom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
NAME_RE = re.compile(r"(IXI\d{3})-([A-Za-z]+)-\d+-T1\.nii\.gz$")


# ------------------------------------------------------------------ download
def download(url, dst):
    if dst.exists():
        return
    print("downloading", url, flush=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    print(f"downloaded {dst.stat().st_size / 1e9:.2f} GB", flush=True)


def extract(tar_path, raw_dir, max_subjects):
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        members = sorted((m for m in tf.getmembers() if NAME_RE.search(m.name)), key=lambda m: m.name)
        if max_subjects:
            members = members[:max_subjects]
        for m in members:
            m.name = Path(m.name).name  # flatten any folder structure
            tf.extract(m, raw_dir)
    print(f"extracted {len(members)} T1 volumes", flush=True)


# ------------------------------------------------------------ skull strip
def run_hdbet(inp, out):
    out.mkdir(parents=True, exist_ok=True)
    help_txt = subprocess.run(["hd-bet", "-h"], capture_output=True, text=True)
    help_txt = help_txt.stdout + help_txt.stderr
    if "disable_tta" in help_txt:  # HD-BET v2
        cmd = ["hd-bet", "-i", str(inp), "-o", str(out), "-device", "cuda", "--disable_tta"]
    else:  # HD-BET v1
        cmd = ["hd-bet", "-i", str(inp), "-o", str(out), "-device", "0", "-mode", "fast", "-tta", "0"]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_stripped(out_dir, stem):
    cands = [p for p in out_dir.glob(f"{stem}*.nii.gz") if "mask" not in p.name]
    return cands[0] if cands else None


# ---------------------------------------------------------- processing
def process(path, voxel):
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    data = zoom(data, [z / voxel for z in zooms], order=1)

    mask = data > 1e-3 * data.max()
    mask = binary_fill_holes(mask)
    lab, n = label(mask)
    if n > 1:  # keep the largest connected component
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == sizes.argmax()

    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    data, mask = data[sl], mask[sl]

    p = np.percentile(data[mask], 99.5)
    data = np.clip(data / p, 0, 1) * mask
    return data.astype(np.float16), mask.astype(np.uint8)


def make_split(id_site, seed, frac=(0.7, 0.1, 0.2)):
    """Subject-level split, stratified by acquisition site."""
    rng = random.Random(seed)
    by_site = {}
    for sid, site in id_site:
        by_site.setdefault(site, []).append(sid)
    out = {"train": [], "val": [], "test": []}
    for site in sorted(by_site):
        ids = sorted(by_site[site])
        rng.shuffle(ids)
        a = round(len(ids) * frac[0])
        b = a + round(len(ids) * frac[1])
        out["train"] += ids[:a]
        out["val"] += ids[a:b]
        out["test"] += ids[b:]
    for k in out:
        out[k].sort()
    return out


def qc_montage(items, path, n=12):
    items = items[:n]
    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, (sid, img, mask) in zip(np.atleast_1d(axs).ravel(), items):
        k = img.shape[2] // 2
        ax.imshow(np.rot90(img[:, :, k].astype(np.float32)), cmap="gray", vmin=0, vmax=1)
        ax.contour(np.rot90(mask[:, :, k]), levels=[0.5], colors="r", linewidths=0.6)
        ax.set_title(sid, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/ixi")
    ap.add_argument("--max_subjects", type=int, default=0, help="0 = all (for smoke tests use ~5)")
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tar_url", default=T1_URL)
    ap.add_argument("--repo", default="", help="Hub dataset repo, e.g. NathanPereira/ixi-t1-prep")
    args = ap.parse_args()

    work = Path(args.work)
    raw, stripped, out = work / "raw", work / "stripped", work / "out"
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    tar_path = work / "IXI-T1.tar"
    download(args.tar_url, tar_path)
    extract(tar_path, raw, args.max_subjects)
    tar_path.unlink()  # free disk space

    run_hdbet(raw, stripped)

    id_site, rows, qc_items, failed = [], [], [], []
    for p in sorted(raw.glob("*.nii.gz")):
        m = NAME_RE.search(p.name)
        sid, site = m.group(1), m.group(2)
        stem = p.name[:-len(".nii.gz")]
        sp = find_stripped(stripped, stem)
        if sp is None:
            failed.append(sid)
            continue
        try:
            img, mask = process(sp, args.voxel)
        except Exception as e:  # keep going; report at the end
            print(f"[{sid}] failed: {e}", flush=True)
            failed.append(sid)
            continue
        np.savez_compressed(out / "volumes" / f"{sid}.npz", image=img, mask=mask)
        id_site.append((sid, site))
        rows.append({"id": sid, "site": site, "shape": "x".join(map(str, img.shape))})
        if len(qc_items) < 12:
            qc_items.append((sid, img, mask))
        print(f"[{sid}] {site} {img.shape}", flush=True)

    splits = make_split(id_site, args.seed)
    split_of = {sid: k for k, ids in splits.items() for sid in ids}
    for r in rows:
        r["split"] = split_of[r["id"]]

    json.dump(splits, open(out / "splits.json", "w"), indent=1)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "site", "shape", "split"])
        w.writeheader()
        w.writerows(rows)
    if qc_items:
        qc_montage(qc_items, out / "qc_montage.png")

    (out / "README.md").write_text(
        "# IXI T1, preprocessed\n\n"
        "Derived from the IXI dataset (brain-development.org/ixi-dataset), licensed CC BY-SA 3.0.\n"
        "Processing: HD-BET skull stripping, RAS reorientation, "
        f"{args.voxel} mm isotropic resampling, crop to brain, 99.5th-percentile normalisation.\n"
        f"Subject-level split stratified by site (seed {args.seed}): "
        + ", ".join(f"{k} {len(v)}" for k, v in splits.items())
        + f".\nFailed subjects: {', '.join(failed) if failed else 'none'}.\n"
    )

    print("split sizes:", {k: len(v) for k, v in splits.items()}, flush=True)
    print("failed:", failed, flush=True)

    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset")
        print("uploaded to", args.repo, flush=True)


if __name__ == "__main__":
    main()
