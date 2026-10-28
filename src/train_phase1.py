# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "monai>=1.3",
#   "huggingface_hub>=0.30",
#   "scikit-image",
#   "scipy",
#   "matplotlib",
#   "numpy",
# ]
# ///
"""
Phase 1: rebuilt baseline for severity-robust-mocorr.

Changes from the ISMRM abstract experiment:
  * data: IXI T1 volumes, skull-stripped, 1 mm isotropic, subject-level split (NathanPereira/ixi-t1-prep)
  * metrics: PSNR / SSIM / NMSE computed inside the brain mask only
  * statistics: per-subject means (n = test subjects), paired Wilcoxon on subjects, not slices
  * an "extreme" test severity that no model sees in training
  * one seed per job; run seeds 0, 1, 2 as separate jobs and aggregate afterwards

Motion model: piecewise-constant rigid in-plane motion along the phase-encode (row) axis.
With 1 mm voxels, translations are in mm.
"""
import argparse, csv, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (max events, max translation [mm], max rotation [deg])
SEVERITY = {
    "mild": (2, 2.0, 2.0),
    "moderate": (4, 4.0, 4.0),
    "severe": (6, 8.0, 8.0),
    "extreme": (8, 12.0, 12.0),  # test only
}
TRAIN_SEVS = {"mild": ["mild"], "mixed": ["mild", "moderate", "severe"]}
TEST_SEVS = ["none", "mild", "moderate", "severe", "extreme"]


# ------------------------------------------------------------------ data
def pad_crop(a, size):
    out = np.zeros((size, size), a.dtype)
    h, w = a.shape
    sh, sw = min(h, size), min(w, size)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    Y0, X0 = (size - sh) // 2, (size - sw) // 2
    out[Y0:Y0 + sh, X0:X0 + sw] = a[y0:y0 + sh, x0:x0 + sw]
    return out


def load_split(root, ids, size, slice_frac, step=1, max_per_subject=0):
    imgs, masks, sids = [], [], []
    for sid in ids:
        d = np.load(root / "volumes" / f"{sid}.npz")
        img, m = d["image"], d["mask"]
        area = m.sum(axis=(0, 1))
        keep = np.where(area >= slice_frac * area.max())[0][::step]
        if max_per_subject and len(keep) > max_per_subject:
            keep = keep[np.unique(np.linspace(0, len(keep) - 1, max_per_subject).round().astype(int))]
        for k in keep:
            imgs.append(pad_crop(img[:, :, k], size))
            masks.append(pad_crop(m[:, :, k], size))
            sids.append(sid)
    return np.stack(imgs).astype(np.float16), np.stack(masks).astype(np.uint8), np.array(sids)


# ------------------------------------------------------- motion simulation
def rigid(x, dx, dy, theta):
    _, _, H, W = x.shape
    c, s = math.cos(theta), math.sin(theta)
    mat = torch.tensor([[c, -s, 2 * dx / W], [s, c, 2 * dy / H]], device=x.device, dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def kspace(x):
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


@torch.no_grad()
def simulate_motion(x, severity, gen):
    if severity == "none":
        return x.clone()
    max_ev, max_t, max_r = SEVERITY[severity]
    B, _, H, _ = x.shape
    out = torch.empty_like(x)
    c0, c1 = int(0.45 * H), int(0.55 * H)
    cand = torch.cat([torch.arange(0, c0), torch.arange(c1, H)])
    for b in range(B):
        n = int(torch.randint(1, max_ev + 1, (1,), generator=gen))
        onsets = cand[torch.randperm(len(cand), generator=gen)[:n]].sort().values.tolist()
        bounds = onsets + [H]
        k = kspace(x[b:b + 1])[0, 0].clone()
        for i in range(n):
            dx, dy = ((torch.rand(2, generator=gen) * 2 - 1) * max_t).tolist()
            th = float((torch.rand(1, generator=gen) * 2 - 1) * math.radians(max_r))
            km = kspace(rigid(x[b:b + 1], dx, dy, th))[0, 0]
            k[bounds[i]:bounds[i + 1]] = km[bounds[i]:bounds[i + 1]]
        out[b, 0] = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs()
    return out.clamp(0, 1.5)


def corrupt_all(X, sev, seed, dev, bs=64):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).float().unsqueeze(1).to(dev)
        outs.append(simulate_motion(x, sev, g).cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------- model
def make_net():
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=(32, 32, 64, 128, 256, 32))


@torch.no_grad()
def predict(net, X, dev, bs=32):
    net.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            y = x + net(x)
        outs.append(y.float().clamp(0, 1).cpu())
    return torch.cat(outs)


def train(mode, Xtr, val_in, val_ref, args, dev):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(1000 * args.seed + {"mild": 1, "mixed": 2}[mode])
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    spe = len(Xtr) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe, pct_start=0.05)
    sevs, t0, log = TRAIN_SEVS[mode], time.time(), []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen).numpy()
        run = 0.0
        for it in range(spe):
            clean = torch.from_numpy(Xtr[perm[it * args.bs:(it + 1) * args.bs]]).float().unsqueeze(1).to(dev)
            sev = sevs[int(torch.randint(len(sevs), (1,), generator=gen))]
            corrupt = simulate_motion(clean, sev, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = corrupt + net(corrupt)
            loss = F.l1_loss(pred.float(), clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        val = F.l1_loss(predict(net, val_in, dev), val_ref).item()
        mins = (time.time() - t0) / 60
        log.append({"epoch": ep + 1, "train_l1": run / spe, "val_l1_moderate": val, "minutes": mins})
        print(f"[{mode} s{args.seed}] epoch {ep + 1}/{args.epochs}  train {run / spe:.4f}  val {val:.4f}  {mins:.1f} min", flush=True)
    return net, log


# --------------------------------------------------------------- metrics
def masked_metrics(pred, ref, mask):
    m = mask.astype(bool)
    d = pred - ref
    mse = float((d[m] ** 2).mean())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("nan")
    _, smap = structural_similarity(ref, pred, data_range=1.0, full=True)
    return psnr, float(smap[m].mean()), float((d[m] ** 2).sum() / ((ref[m] ** 2).sum() + 1e-12))


def evaluate(pred, ref, masks, sids):
    p, r = pred.numpy()[:, 0], ref.numpy()[:, 0]
    rows = []
    for i in range(len(p)):
        ps, ss, nm = masked_metrics(p[i], r[i], masks[i])
        rows.append({"sid": sids[i], "psnr": ps, "ssim": ss, "nmse": nm})
    return rows


def per_subject(rows):
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    return {sid: {k: float(np.nanmean([x[k] for x in rs])) for k in ("psnr", "ssim", "nmse")} for sid, rs in by.items()}


def stat(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))} if len(v) else {"mean": None, "std": None, "n": 0}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_repo", default="NathanPereira/ixi-t1-prep")
    ap.add_argument("--size", type=int, default=208)
    ap.add_argument("--slice_frac", type=float, default=0.2, help="keep slices with brain area >= frac * max")
    ap.add_argument("--train_step", type=int, default=3, help="use every n-th training slice")
    ap.add_argument("--test_per_subject", type=int, default=30)
    ap.add_argument("--max_train_subjects", type=int, default=0, help="0 = all (smoke tests: ~10)")
    ap.add_argument("--max_test_subjects", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repo", default="", help="Hub repo for results, e.g. NathanPereira/mocorr-phase1")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(args.data_repo, repo_type="dataset"))
    splits = json.load(open(root / "splits.json"))
    tr_ids = splits["train"][: args.max_train_subjects or None]
    va_ids = splits["val"][: max(2, (args.max_train_subjects or 0) // 4) if args.max_train_subjects else None]
    te_ids = splits["test"][: args.max_test_subjects or None]

    Xtr, _, _ = load_split(root, tr_ids, args.size, args.slice_frac, step=args.train_step)
    Xva, _, _ = load_split(root, va_ids, args.size, args.slice_frac, max_per_subject=8)
    Xte, Mte, Ste = load_split(root, te_ids, args.size, args.slice_frac, max_per_subject=args.test_per_subject)
    print(f"device {dev} | train {len(tr_ids)} subj / {len(Xtr)} slices | val {len(va_ids)} / {len(Xva)} | "
          f"test {len(te_ids)} / {len(Xte)}", flush=True)

    val_ref = torch.from_numpy(Xva).float().unsqueeze(1)
    val_in = corrupt_all(Xva, "moderate", 999, dev)
    ref_te = torch.from_numpy(Xte).float().unsqueeze(1)
    test_in = {sev: corrupt_all(Xte, sev, 1234, dev) for sev in TEST_SEVS}  # identical across seeds and models

    models, logs = {}, {}
    for mode in ["mild", "mixed"]:
        models[mode], logs[mode] = train(mode, Xtr, val_in, val_ref, args, dev)
        torch.save(models[mode].state_dict(), out / f"unet_{mode}.pt")

    summary, per_slice_rows, per_subj_rows, preds = {}, [], [], {}
    for sev in TEST_SEVS:
        methods = {"input": test_in[sev]}
        for mode, net in models.items():
            preds[(mode, sev)] = methods[mode] = predict(net, test_in[sev], dev)
        summary[sev] = {}
        subj = {}
        for name, pred in methods.items():
            rows = evaluate(pred, ref_te, Mte, Ste)
            subj[name] = per_subject(rows)
            summary[sev][name] = {
                "slice_level": {k: stat([r[k] for r in rows]) for k in ("psnr", "ssim", "nmse")},
                "subject_level": {k: stat([v[k] for v in subj[name].values()]) for k in ("psnr", "ssim", "nmse")},
            }
            per_slice_rows += [{"severity": sev, "method": name, **r} for r in rows]
            per_subj_rows += [{"severity": sev, "method": name, "sid": s, **v} for s, v in subj[name].items()]
        if sev != "none":
            ids = sorted(subj["mild"])
            for k in ("psnr", "ssim"):
                a = [subj["mixed"][s][k] for s in ids]
                b = [subj["mild"][s][k] for s in ids]
                summary[sev][f"wilcoxon_subject_{k}_mixed_vs_mild_p"] = float(wilcoxon(a, b).pvalue) if len(ids) > 5 else None

    json.dump({"args": vars(args), "n_test_subjects": len(te_ids), "train_logs": logs, "summary": summary},
              open(out / "results.json", "w"), indent=2)
    for name, rows in (("per_slice_metrics.csv", per_slice_rows), ("per_subject_metrics.csv", per_subj_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # figure: subject-level metrics vs severity
    labels = {"input": "Motion-corrupted", "mild": "U-Net (mild-trained)", "mixed": "U-Net (mixed-trained)"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, k, name in zip(axs, ["psnr", "ssim"], ["Brain PSNR (dB)", "Brain SSIM"]):
        for m in ["input", "mild", "mixed"]:
            pts = [(i, summary[s][m]["subject_level"][k]) for i, s in enumerate(TEST_SEVS)
                   if summary[s][m]["subject_level"][k]["mean"] is not None]
            ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts], yerr=[p[1]["std"] for p in pts],
                        marker="o", capsize=3, label=labels[m])
        ax.axvspan(3.5, 4.5, color="0.9", zorder=0)
        ax.set_xticks(range(len(TEST_SEVS)))
        ax.set_xticklabels(TEST_SEVS)
        ax.set_ylabel(name)
        ax.set_xlabel("Test motion severity (shaded: unseen in training)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_metrics_vs_severity.png", dpi=300)

    # figure: example slice
    i = len(Xte) // 2
    sevs = ["mild", "moderate", "severe", "extreme"]
    cols = ["Reference", "Corrupted", "Mild-trained", "Mixed-trained", "|Error| mixed ×5"]
    fig, axs = plt.subplots(len(sevs), 5, figsize=(12, 2.6 * len(sevs)))
    for r, sev in enumerate(sevs):
        ims = [ref_te[i, 0], test_in[sev][i, 0], preds[("mild", sev)][i, 0], preds[("mixed", sev)][i, 0],
               (preds[("mixed", sev)][i, 0] - ref_te[i, 0]).abs() * 5 * torch.from_numpy(Mte[i]).float()]
        for c, im in enumerate(ims):
            axs[r, c].imshow(np.rot90(im.numpy()), cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=10)
        axs[r, 0].text(-8, args.size / 2, sev, rotation=90, va="center", ha="right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_examples.png", dpi=200)

    print(json.dumps({s: {m: summary[s][m]["subject_level"]["psnr"]["mean"] for m in ("input", "mild", "mixed")}
                      for s in TEST_SEVS}, indent=1), flush=True)

    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model", path_in_repo=f"seed{args.seed}")
        print(f"uploaded to {args.repo}/seed{args.seed}", flush=True)


if __name__ == "__main__":
    main()
