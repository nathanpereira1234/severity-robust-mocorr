# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.30", "pandas", "numpy", "scipy", "matplotlib"]
# ///
"""
Aggregate Phase 2 results from <repo>/seed*/<tag>/per_subject.csv.

Produces (uploaded to <repo>/aggregate/):
  table_model_psnr.csv     mean (over seeds) of subject-level PSNR per training range x test condition
  table_gain.csv           same, minus the corrupted-input PSNR
  stats_vs_reference.csv   paired Wilcoxon (subjects) of every model vs the reference model on key conditions,
                           Bonferroni-corrected
  fig_heatmap.png          training range (rows) x test condition (columns), PSNR gain over input
  fig_magnitude_effect.png test extent fixed at 100%, PSNR vs test magnitude, one line per training magnitude
  fig_extent_effect.png    test magnitude fixed at 8, PSNR vs test extent, one line per training extent
  fig_cost.png             clean and mild-motion PSNR vs training magnitude, one line per training extent
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_tag(tag):
    parts = tag.split("_")
    return float(parts[0][1:]), int(parts[1][1:]) / 100, "_".join(parts[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="NathanPereira/mocorr-phase2")
    ap.add_argument("--metric", default="psnr_reg", help="psnr_reg (default), psnr, ssim_reg, ssim")
    ap.add_argument("--reference", default="m12_e100", help="model tag that others are compared against")
    ap.add_argument("--out", default="aggregate")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download, HfApi
    root = Path(snapshot_download(args.repo, allow_patterns=["seed*/*/per_subject.csv"]))
    frames = []
    for p in sorted(root.glob("seed*/*/per_subject.csv")):
        d = pd.read_csv(p)
        d["seed"] = int(p.parent.parent.name[4:])
        d["tag"] = p.parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    tags = sorted(df.tag.unique(), key=lambda t: parse_tag(t)[:2])
    conds = list(dict.fromkeys(df.cond))
    print("models:", tags, "\nseeds:", sorted(df.seed.unique()), "\nconditions:", conds)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    met = args.metric

    # mean over subjects, then mean/SD over seeds
    by_seed = df.groupby(["tag", "seed", "cond", "method"])[met].mean().reset_index()
    model = by_seed[by_seed.method == "model"].groupby(["tag", "cond"])[met].agg(["mean", "std"]).reset_index()
    inp = by_seed[by_seed.method == "input"].groupby("cond")[met].mean()

    tbl = model.pivot(index="tag", columns="cond", values="mean").loc[tags, conds]
    tbl.to_csv(out / "table_model_psnr.csv")
    model.pivot(index="tag", columns="cond", values="std").loc[tags, conds].to_csv(out / "table_model_sd.csv")
    gain = tbl.sub(inp.reindex(conds), axis=1)
    gain.to_csv(out / "table_gain.csv")

    # stats: every model vs reference on key conditions (subject level, seed 0 or first seed)
    key = [c for c in ["none", "m2_e025", "m4_e050", "m12_e100", "m16_e100"] if c in conds]
    first = sorted(df.seed.unique())[0]
    d0 = df[(df.seed == first) & (df.method == "model")]
    others = [t for t in tags if t != args.reference]
    n_tests = len(others) * len(key)
    rows = []
    for c in key:
        ref = d0[(d0.tag == args.reference) & (d0.cond == c)].set_index("sid")[met]
        for t in others:
            x = d0[(d0.tag == t) & (d0.cond == c)].set_index("sid")[met]
            ids = ref.index.intersection(x.index)
            a, b = x.loc[ids].values, ref.loc[ids].values
            ok = np.isfinite(a) & np.isfinite(b)
            p = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() > 5 and np.any(a[ok] != b[ok]) else np.nan
            rows.append({"cond": c, "model": t, "reference": args.reference, "mean_diff": float(np.mean(a[ok] - b[ok])),
                         "p": p, "p_bonferroni": min(1.0, p * n_tests) if np.isfinite(p) else np.nan})
    pd.DataFrame(rows).to_csv(out / "stats_vs_reference.csv", index=False)

    # heatmap
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(conds), 0.9 + 0.45 * len(tags)))
    im = ax.imshow(gain.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=8)
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            v = gain.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="w")
    ax.set_xlabel("Test condition (magnitude_extent)")
    ax.set_ylabel("Training range")
    fig.colorbar(im, ax=ax, label=f"{met} gain over input (dB)" if "psnr" in met else f"{met} gain")
    fig.tight_layout()
    fig.savefig(out / "fig_heatmap.png", dpi=200)

    meta = pd.DataFrame([dict(tag=t, train_mag=parse_tag(t)[0], train_ext=parse_tag(t)[1]) for t in tags]).set_index("tag")

    def line_plot(fname, fixed_ext_tr, test_conds, xvals, xlabel, group_col, group_vals, title):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(xvals, [inp.get(c, np.nan) for c in test_conds], "k--", label="input")
        for gv in group_vals:
            sel = meta[(meta[group_col] == gv) & ((meta.train_ext == fixed_ext_tr) if group_col == "train_mag"
                                                   else (meta.train_mag == fixed_ext_tr))]
            for t in sel.index:
                ax.plot(xvals, [tbl.loc[t, c] if c in tbl.columns else np.nan for c in test_conds], marker="o", label=t)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(met)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)

    mags = [2, 4, 8, 12, 16]
    ext_conds = [f"m{m}_e100" for m in mags]
    line_plot("fig_magnitude_effect.png", 1.0, ext_conds, mags, "Test magnitude (mm / deg), test extent 100%",
              "train_mag", sorted(meta.train_mag.unique()), "Magnitude effect (training extent 100%)")
    exts = [0.25, 0.5, 1.0]
    mag_conds = [f"m8_e{int(e * 100):03d}" for e in exts]
    line_plot("fig_extent_effect.png", 8.0, mag_conds, exts, "Test extent (fraction of k-space), test magnitude 8",
              "train_ext", sorted(meta.train_ext.unique()), "Extent effect (training magnitude 8)")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, c, ttl in zip(axs, ["none", "m2_e025"], ["Clean images", "Mild motion (m2, 25%)"]):
        for e in sorted(meta.train_ext.unique()):
            sel = meta[meta.train_ext == e].sort_values("train_mag")
            ax.plot(sel.train_mag, [tbl.loc[t, c] for t in sel.index], marker="o", label=f"train extent {int(e * 100)}%")
        ax.set_xlabel("Training magnitude (mm / deg)")
        ax.set_ylabel(met)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_cost.png", dpi=200)

    print(gain.round(2).to_string())
    if args.upload:
        HfApi().upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model", path_in_repo="aggregate")
        print("uploaded to", args.repo + "/aggregate")


if __name__ == "__main__":
    main()
