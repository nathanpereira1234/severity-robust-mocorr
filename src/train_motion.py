# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "monai>=1.3",
#   "datasets>=2.19",
#   "huggingface_hub>=0.30",
#   "scikit-image",
#   "scipy",
#   "matplotlib",
#   "numpy",
#   "pillow",
# ]
# ///
"""
ISMRM 2027 abstract experiment
Does a deep-learning retrospective motion-correction network generalise across
motion severities? Rigid in-plane motion is simulated in k-space on IXI2D slices.

Models:
  mild   - trained on mild motion only
  mixed  - trained on mild + moderate + severe
Test:    none / mild / moderate / severe (fixed seed, identical inputs for both models)

Outputs (also pushed to a private Hub repo): results.json, per_image_metrics.csv,
fig_metrics_vs_severity.png, fig_examples.png, model weights.
"""
import argparse, csv, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from monai.networks.nets import BasicUNet
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (max motion events, max translation [px], max rotation [deg])
SEVERITY = {
    "mild": (2, 2.0, 2.0),
    "moderate": (4, 4.0, 4.0),
    "severe": (6, 8.0, 8.0),
}
TRAIN_SEVS = {"mild": ["mild"], "mixed": ["mild", "moderate", "severe"]}
TEST_SEVS = ["none", "mild", "moderate", "severe"]


# ----------------------------------------------------------------- data
PARQUET = ("https://huggingface.co/datasets/iamkzntsv/IXI2D/resolve/"
           "refs%2Fconvert%2Fparquet/default/{split}/0000.parquet")


def _to_pil(im):
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, dict):  # raw {"bytes": ..., "path": ...}
        import io
        return Image.open(io.BytesIO(im["bytes"]))
    raise TypeError(f"unexpected image type {type(im)}")


def load_split(split, img, n_max):
    # Load the Hub's auto-converted parquet; the raw zip loader breaks on the
    # validation folder's class labels.
    ds = load_dataset("parquet", data_files={split: PARQUET.format(split=split)}, split=split)
    if n_max and n_max < len(ds):
        ds = ds.select(range(n_max))
    arr = np.stack([
        np.asarray(_to_pil(im).convert("L").resize((img, img), Image.BILINEAR), dtype=np.uint8)
        for im in ds["image"]
    ])
    return arr  # (N, img, img) uint8


# ------------------------------------------------------ motion simulation
def rigid(x, dx, dy, theta):
    """x: (1,1,H,W); dx, dy in pixels; theta in radians (floats)."""
    _, _, H, W = x.shape
    c, s = math.cos(theta), math.sin(theta)
    mat = torch.tensor([[c, -s, 2 * dx / W], [s, c, 2 * dy / H]],
                       device=x.device, dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def kspace(x):
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


@torch.no_grad()
def simulate_motion(x, severity, gen):
    """Piecewise-constant rigid motion along the phase-encode (row) direction.
    Each motion event changes the pose for all subsequently acquired k-space lines."""
    if severity == "none":
        return x.clone()
    max_ev, max_t, max_r = SEVERITY[severity]
    B, _, H, _ = x.shape
    out = torch.empty_like(x)
    c0, c1 = int(0.45 * H), int(0.55 * H)  # don't let an event split the k-space centre
    cand = torch.cat([torch.arange(0, c0), torch.arange(c1, H)])
    for b in range(B):
        n = int(torch.randint(1, max_ev + 1, (1,), generator=gen))
        onsets = cand[torch.randperm(len(cand), generator=gen)[:n]].sort().values.tolist()
        bounds = onsets + [H]
        k = kspace(x[b:b + 1])[0, 0].clone()
        for i in range(n):
            dx, dy = ((torch.rand(2, generator=gen) * 2 - 1) * max_t).tolist()
            th = float((torch.rand(1, generator=gen) * 2 - 1) * math.radians(max_r))
            moved = rigid(x[b:b + 1], dx, dy, th)
            km = kspace(moved)[0, 0]
            k[bounds[i]:bounds[i + 1]] = km[bounds[i]:bounds[i + 1]]
        out[b, 0] = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs()
    return out.clamp(0, 1.5)


# ---------------------------------------------------------------- model
def make_net():
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1,
                     features=(32, 32, 64, 128, 256, 32))


def to_batch(arr, dev):
    return torch.from_numpy(arr).to(dev).float().div(255.0).unsqueeze(1)


def train(mode, Xtr, args, dev):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + {"mild": 1, "mixed": 2}[mode])
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    spe = len(Xtr) // args.bs
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    sevs, t0, step = TRAIN_SEVS[mode], time.time(), 0
    log = []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen).numpy()
        run = 0.0
        for it in range(spe):
            clean = to_batch(Xtr[perm[it * args.bs:(it + 1) * args.bs]], dev)
            sev = sevs[int(torch.randint(len(sevs), (1,), generator=gen))]
            corrupt = simulate_motion(clean, sev, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = corrupt + net(corrupt)  # residual learning
            loss = F.l1_loss(pred.float(), clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
        mins = (time.time() - t0) / 60
        log.append({"epoch": ep + 1, "l1": run / spe, "minutes": mins})
        print(f"[{mode}] epoch {ep + 1}/{args.epochs}  L1 {run / spe:.4f}  {mins:.1f} min", flush=True)
        if mins > args.max_minutes:
            print(f"[{mode}] time budget reached, stopping", flush=True)
            break
    return net, log


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


def metrics(pred, ref):
    p, r = pred.numpy()[:, 0], ref.numpy()[:, 0]
    res = []
    for a, b in zip(p, r):
        a = np.clip(a, 0, 1)
        identical = np.array_equal(a, b)  # motion-free input vs reference
        res.append({
            "psnr": float("nan") if identical else peak_signal_noise_ratio(b, a, data_range=1.0),
            "ssim": structural_similarity(b, a, data_range=1.0),
            "nmse": float(np.sum((a - b) ** 2) / (np.sum(b ** 2) + 1e-12)),
        })
    return res


def _stat(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "std": None}
    return {"mean": float(v.mean()), "std": float(v.std())}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all slices")
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--max_minutes", type=float, default=80, help="per model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repo", default="", help="Hub repo to push results, e.g. NathanPereira/ismrm27-motion")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print("device:", dev, flush=True)

    Xtr = load_split("train", args.img, args.n_train)
    Xte = load_split("validation", args.img, args.n_test)
    print(f"train {Xtr.shape}  test {Xte.shape}", flush=True)

    # fixed, shared test corruptions
    clean_te = to_batch(Xte, "cpu")
    test_in = {}
    for sev in TEST_SEVS:
        g = torch.Generator().manual_seed(1234)
        chunks = [simulate_motion(clean_te[i:i + 64].to(dev), sev, g).cpu()
                  for i in range(0, len(clean_te), 64)]
        test_in[sev] = torch.cat(chunks)

    models, logs = {}, {}
    for mode in ["mild", "mixed"]:
        models[mode], logs[mode] = train(mode, Xtr, args, dev)
        torch.save(models[mode].state_dict(), f"{args.out}/unet_{mode}.pt")

    # evaluation
    per_img, summary, preds = [], {}, {}
    for sev in TEST_SEVS:
        rows = {"input": metrics(test_in[sev], clean_te)}
        for mode, net in models.items():
            preds[(mode, sev)] = predict(net, test_in[sev], dev)
            rows[mode] = metrics(preds[(mode, sev)], clean_te)
        summary[sev] = {}
        for method, ms in rows.items():
            summary[sev][method] = {
                k: _stat([m[k] for m in ms])
                for k in ("psnr", "ssim", "nmse")
            }
            for i, m in enumerate(ms):
                per_img.append({"severity": sev, "method": method, "idx": i, **m})
        # paired test mixed vs mild
        if sev != "none":
            a = [m["psnr"] for m in rows["mixed"]]
            b = [m["psnr"] for m in rows["mild"]]
            summary[sev]["wilcoxon_psnr_mixed_vs_mild_p"] = float(wilcoxon(a, b).pvalue)

    json.dump({"args": vars(args), "train_logs": logs, "summary": summary},
              open(f"{args.out}/results.json", "w"), indent=2)
    with open(f"{args.out}/per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys()))
        w.writeheader()
        w.writerows(per_img)

    # figure 1: metrics vs severity
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    labels = {"input": "Motion-corrupted", "mild": "U-Net (mild-trained)", "mixed": "U-Net (mixed-trained)"}
    for ax, k, name in zip(axs, ["psnr", "ssim"], ["PSNR (dB)", "SSIM"]):
        for method in ["input", "mild", "mixed"]:
            pts = [(s, summary[s][method][k]) for s in TEST_SEVS
                   if summary[s][method][k]["mean"] is not None]
            ax.errorbar([TEST_SEVS.index(p[0]) for p in pts], [p[1]["mean"] for p in pts],
                        yerr=[p[1]["std"] for p in pts], marker="o", capsize=3, label=labels[method])
        ax.set_xticks(range(len(TEST_SEVS)))
        ax.set_xticklabels(TEST_SEVS)
        ax.set_ylabel(name)
        ax.set_xlabel("Test motion severity")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_metrics_vs_severity.png", dpi=300)

    # figure 2: example slice
    i = int(len(clean_te) // 2)
    cols = ["Reference", "Corrupted", "Mild-trained", "Mixed-trained", "|Error| mixed ×5"]
    fig, axs = plt.subplots(3, 5, figsize=(12, 7.5))
    for r, sev in enumerate(["mild", "moderate", "severe"]):
        ims = [clean_te[i, 0], test_in[sev][i, 0], preds[("mild", sev)][i, 0],
               preds[("mixed", sev)][i, 0],
               (preds[("mixed", sev)][i, 0] - clean_te[i, 0]).abs() * 5]
        for c, im in enumerate(ims):
            axs[r, c].imshow(im.numpy(), cmap="gray", vmin=0, vmax=1)
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=10)
        axs[r, 0].text(-10, args.img / 2, sev, rotation=90, va="center", ha="right", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_examples.png", dpi=300)

    print(json.dumps(summary, indent=2), flush=True)

    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.repo, repo_type="model")
        print("uploaded to", args.repo, flush=True)


if __name__ == "__main__":
    main()