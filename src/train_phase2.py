# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "monai>=1.3",
#   "huggingface_hub>=0.30",
#   "scikit-image",
#   "scipy",
#   "numpy",
# ]
# ///
"""
Phase 2: which dimension of motion severity drives robustness?

Each job trains ONE network on a training range defined by
  --mag M      maximum motion magnitude (mm and degrees); each sample draws m ~ U(0, M)
  --extent E   maximum fraction of phase-encode lines acquired while displaced; each sample draws e ~ U(0.05, E)
and evaluates it on a fixed grid of test conditions (magnitude x extent) plus clean images.

Motion model: inside a window of e*H consecutive phase-encode lines (random position, k-space centre allowed),
the head takes 1..n_max piecewise-constant rigid poses (|t| <= m mm, |rot| <= m deg); outside the window it is at
the reference pose. Optional --motion walk uses smooth random-walk poses per 8-line segment plus sudden jumps.

Metrics: brain-masked PSNR / SSIM / NMSE per subject, both raw and after translation co-registration of each
image to the reference (phase correlation), following Reguig et al. 2022.
Outputs go to <repo>/seed<s>/<tag>/ where tag = m<M>_e<E*100>.
"""
import argparse, csv, gc, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import BasicUNet
from scipy.ndimage import shift as nd_shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

TEST_MAGS = [2, 4, 8, 12, 16]          # 16 is beyond every training range
TEST_EXTENTS = [0.25, 0.5, 1.0]
VAL_COND = (4, 0.5)


# ------------------------------------------------------------------ data (as Phase 1)
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


# ------------------------------------------------------------------ motion simulation
def rigid(x, dx, dy, theta):
    _, _, H, W = x.shape
    c, s = math.cos(theta), math.sin(theta)
    mat = torch.tensor([[c, -s, 2 * dx / W], [s, c, 2 * dy / H]], device=x.device, dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def kspace(x):
    return torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))


def rand_pose(m, gen):
    dx, dy, th = ((torch.rand(3, generator=gen) * 2 - 1) * m).tolist()
    return dx, dy, math.radians(th)


def poses_for_window(L, m, mode, n_max, gen, seg=8):
    """Return a list of (start, end, pose) segments covering [0, L)."""
    if mode == "walk":
        nseg = max(1, math.ceil(L / seg))
        steps = (torch.randn(nseg, 3, generator=gen) * 0.35).cumsum(0)          # smooth drift
        jumps = (torch.rand(nseg, generator=gen) < 0.1).float().unsqueeze(1) * (torch.rand(nseg, 3, generator=gen) * 2 - 1) * 1.5
        traj = steps + jumps.cumsum(0)
        traj = traj / traj.abs().max().clamp(min=1e-6) * m                      # scale to magnitude m
        out = []
        for i in range(nseg):
            a, b = i * seg, min(L, (i + 1) * seg)
            dx, dy, th = traj[i].tolist()
            out.append((a, b, (dx, dy, math.radians(th))))
        return out
    n = int(torch.randint(1, n_max + 1, (1,), generator=gen))
    cuts = sorted(torch.randperm(max(L - 1, 1), generator=gen)[: n - 1].add(1).tolist()) if n > 1 and L > 1 else []
    bounds = [0] + cuts + [L]
    return [(bounds[i], bounds[i + 1], rand_pose(m, gen)) for i in range(len(bounds) - 1)]


@torch.no_grad()
def simulate(x, m, e, gen, mode="piecewise", n_max=6):
    """x: (B,1,H,W). m: magnitude (mm & deg); e: fraction of PE lines displaced. m or e == 0 -> clean."""
    if m <= 0 or e <= 0:
        return x.clone()
    B, _, H, _ = x.shape
    L = max(1, int(round(e * H)))
    out = torch.empty_like(x)
    for b in range(B):
        start = int(torch.randint(0, H - L + 1, (1,), generator=gen))
        k = kspace(x[b:b + 1])[0, 0].clone()
        for a, c, (dx, dy, th) in poses_for_window(L, m, mode, n_max, gen):
            km = kspace(rigid(x[b:b + 1], dx, dy, th))[0, 0]
            k[start + a:start + c] = km[start + a:start + c]
        out[b, 0] = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs()
    return out.clamp(0, 1.5)


def corrupt_all(X, m, e, seed, dev, mode, bs=64):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).float().unsqueeze(1).to(dev)
        outs.append(simulate(x, m, e, g, mode).half().cpu())
    return torch.cat(outs)


# ------------------------------------------------------------------ model / training
def make_net(width=1):
    f = tuple(int(c * width) for c in (32, 32, 64, 128, 256, 32))
    return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=f)


@torch.no_grad()
def predict(net, X, dev, bs=32):
    net.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            y = x + net(x)
        outs.append(y.float().clamp(0, 1).half().cpu())
    return torch.cat(outs)


def train(Xtr, val_in, val_ref, args, dev):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(10_000 * args.seed + int(args.mag * 100) + int(args.extent * 1000))
    net = make_net(args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    spe = len(Xtr) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe, pct_start=0.05)
    t0, log = time.time(), []
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen).numpy()
        run = 0.0
        for it in range(spe):
            clean = torch.from_numpy(Xtr[perm[it * args.bs:(it + 1) * args.bs]]).float().unsqueeze(1).to(dev)
            m = float(torch.rand(1, generator=gen)) * args.mag
            e = 0.05 + float(torch.rand(1, generator=gen)) * max(args.extent - 0.05, 0.0)
            corrupt = simulate(clean, m, e, gen, args.motion)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = corrupt + net(corrupt)
            loss = F.l1_loss(pred.float(), clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        val = F.l1_loss(predict(net, val_in, dev).float(), val_ref).item()
        mins = (time.time() - t0) / 60
        log.append({"epoch": ep + 1, "train_l1": run / spe, "val_l1": val, "minutes": mins})
        print(f"[{args.tag} s{args.seed}] epoch {ep + 1}/{args.epochs}  train {run / spe:.4f}  val {val:.4f}  {mins:.1f} min",
              flush=True)
    return net, log


# ------------------------------------------------------------------ metrics
def masked(pred, ref, mask):
    m = mask.astype(bool)
    d = pred - ref
    mse = float((d[m] ** 2).mean())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("nan")
    _, smap = structural_similarity(ref, pred, data_range=1.0, full=True)
    return psnr, float(smap[m].mean()), float((d[m] ** 2).sum() / ((ref[m] ** 2).sum() + 1e-12))


def register(img, ref):
    if np.array_equal(img, ref):
        return img
    sh, _, _ = phase_cross_correlation(ref, img, upsample_factor=10, normalization=None)
    return nd_shift(img, sh, order=1, mode="constant")


def evaluate(pred, ref, masks, sids):
    p, r = pred.numpy()[:, 0].astype(np.float32), ref.numpy()[:, 0].astype(np.float32)
    rows = []
    for i in range(len(p)):
        raw = masked(p[i], r[i], masks[i])
        reg = masked(register(p[i], r[i]), r[i], masks[i])
        rows.append({"sid": sids[i], "psnr": raw[0], "ssim": raw[1], "nmse": raw[2],
                     "psnr_reg": reg[0], "ssim_reg": reg[1], "nmse_reg": reg[2]})
    return rows


def per_subject(rows):
    keys = ("psnr", "ssim", "nmse", "psnr_reg", "ssim_reg", "nmse_reg")
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    out = {}
    for sid, rs in by.items():
        out[sid] = {}
        for k in keys:
            v = np.array([x[k] for x in rs], float)
            v = v[np.isfinite(v)]
            out[sid][k] = float(v.mean()) if len(v) else float("nan")
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag", type=float, required=True, help="max training magnitude (mm and degrees)")
    ap.add_argument("--extent", type=float, required=True, help="max training extent (fraction of PE lines)")
    ap.add_argument("--motion", choices=["piecewise", "walk"], default="piecewise")
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier (2 = capacity test)")
    ap.add_argument("--data_repo", default="NathanPereira/ixi-t1-prep")
    ap.add_argument("--size", type=int, default=208)
    ap.add_argument("--slice_frac", type=float, default=0.2)
    ap.add_argument("--train_step", type=int, default=3)
    ap.add_argument("--test_per_subject", type=int, default=15)
    ap.add_argument("--max_train_subjects", type=int, default=0)
    ap.add_argument("--max_test_subjects", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repo", default="", help="e.g. NathanPereira/mocorr-phase2")
    args = ap.parse_args()
    args.tag = f"m{args.mag:g}_e{int(round(args.extent * 100)):03d}" + ("_walk" if args.motion == "walk" else "") \
               + (f"_w{args.width:g}" if args.width != 1 else "")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / f"seed{args.seed}" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download, HfApi
    root = Path(snapshot_download(args.data_repo, repo_type="dataset"))
    splits = json.load(open(root / "splits.json"))
    tr_ids = splits["train"][: args.max_train_subjects or None]
    va_ids = splits["val"][: 4 if args.max_train_subjects else None]
    te_ids = splits["test"][: args.max_test_subjects or None]
    Xtr, _, _ = load_split(root, tr_ids, args.size, args.slice_frac, step=args.train_step)
    Xva, _, _ = load_split(root, va_ids, args.size, args.slice_frac, max_per_subject=8)
    Xte, Mte, Ste = load_split(root, te_ids, args.size, args.slice_frac, max_per_subject=args.test_per_subject)
    print(f"[{args.tag}] device {dev} | train {len(tr_ids)}/{len(Xtr)} | val {len(va_ids)}/{len(Xva)} | "
          f"test {len(te_ids)}/{len(Xte)}", flush=True)

    val_ref = torch.from_numpy(Xva).float().unsqueeze(1)
    val_in = corrupt_all(Xva, *VAL_COND, 999, dev, "piecewise")

    api = None
    if args.repo:
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)

    net, log = train(Xtr, val_in, val_ref, args, dev)
    torch.save(net.state_dict(), out / "unet.pt")
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
    if api:
        api.upload_file(path_or_fileobj=str(out / "unet.pt"), repo_id=args.repo, repo_type="model",
                        path_in_repo=f"seed{args.seed}/{args.tag}/unet.pt")
    del Xtr
    gc.collect()

    ref = torch.from_numpy(Xte).unsqueeze(1)
    conds = [("none", 0, 0.0)] + [(f"m{m}_e{int(e * 100):03d}", m, e) for m in TEST_MAGS for e in TEST_EXTENTS]
    summary, subj_rows = {}, []
    for ci, (name, m, e) in enumerate(conds):
        t = time.time()
        corrupted = corrupt_all(Xte, m, e, 1234 + ci, dev, "piecewise")   # identical across all Phase 2 models
        pred = predict(net, corrupted, dev)
        res = {}
        for method, imgs in (("input", corrupted), ("model", pred)):
            subj = per_subject(evaluate(imgs, ref, Mte, Ste))
            res[method] = {k: float(np.nanmean([v[k] for v in subj.values()])) for k in next(iter(subj.values()))}
            subj_rows += [{"cond": name, "test_mag": m, "test_extent": e, "method": method, "sid": s, **v}
                          for s, v in subj.items()]
        summary[name] = res
        print(f"[{args.tag}] {name:10s} input {res['input']['psnr_reg']:.2f}  model {res['model']['psnr_reg']:.2f} dB "
              f"(registered)  {time.time() - t:.0f}s", flush=True)
        del corrupted, pred
        gc.collect()

    json.dump({"args": vars(args), "train_log": log, "summary": summary,
               "test_mags": TEST_MAGS, "test_extents": TEST_EXTENTS}, open(out / "results.json", "w"), indent=1)
    with open(out / "per_subject.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(subj_rows[0].keys()))
        w.writeheader()
        w.writerows(subj_rows)
    if api:
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model",
                          path_in_repo=f"seed{args.seed}/{args.tag}")
        print(f"uploaded to {args.repo}/seed{args.seed}/{args.tag}", flush=True)


if __name__ == "__main__":
    main()
