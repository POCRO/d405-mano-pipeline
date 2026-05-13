"""
Compare FoundationStereo depth with D405's native depth on the same capture.

Usage:
  python compare_fs_d405.py captures/<timestamp>_emON
  python compare_fs_d405.py --latest      # use the most recent capture
  python compare_fs_d405.py captures/* --batch   # all captures

Pipeline:
  1. Load left.png, right.png, K.txt, depth_d405.npy from the capture dir
  2. Run FoundationStereo (ViT-L) to produce fs_depth.npy in the same frame
  3. Save a 4-panel comparison.png + comparison_stats.txt under the capture dir

The captures coordinate system is the left-IR camera frame. D405's native depth
stream is already in that frame, so no spatial alignment is needed. Both depth
maps are float32 meters.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Where the FoundationStereo repo lives — adjust if you move things.
FS_ROOT = Path(__file__).resolve().parent.parent / "FoundationStereo"
CKPT = FS_ROOT / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
CFG = FS_ROOT / "pretrained_models" / "23-51-11" / "cfg.yaml"

Z_MIN, Z_MAX = 0.05, 1.5  # match capture_stereo.py


def colorize_depth(depth_m, z_min=Z_MIN, z_max=Z_MAX):
    valid = (depth_m > z_min) & (depth_m < z_max)
    norm = np.zeros_like(depth_m, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, z_min, z_max)
        norm = ((clipped - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def colorize_diff(diff_m, abs_max=0.05):
    """Diff in meters -> BGR. Red = FS deeper, blue = FS closer, white = match."""
    norm = np.clip(diff_m / abs_max, -1.0, 1.0)
    mag = np.abs(norm) * 255.0
    pos = (norm > 0)
    vis = np.full((*diff_m.shape, 3), 255, dtype=np.uint8)  # start white
    # OpenCV uses BGR. Red = (0,0,255), Blue = (255,0,0).
    # As |norm| grows, fade white toward saturated color.
    fade = (255 - mag).astype(np.uint8)
    vis[..., 0] = np.where(pos, fade, 255)              # B
    vis[..., 1] = fade.astype(np.uint8)                  # G shrinks both ways
    vis[..., 2] = np.where(pos, 255, fade)              # R
    return vis


def load_fs_model():
    """Lazy import + load. Returns (model, args) tuple."""
    sys.path.insert(0, str(FS_ROOT))
    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(CFG)
    cfg["vit_size"] = cfg.get("vit_size", "vitl")
    cfg["valid_iters"] = 32
    cfg["hiera"] = 0
    cfg["low_memory"] = 0
    cfg["mixed_precision"] = True
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    torch.autograd.set_grad_enabled(False)
    return model, args


def run_fs(model, args, left_path, right_path, K_path):
    import imageio
    import torch
    sys.path.insert(0, str(FS_ROOT))
    from core.utils.utils import InputPadder

    img0 = imageio.imread(str(left_path))
    img1 = imageio.imread(str(right_path))
    H, W = img0.shape[:2]
    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)
    with torch.cuda.amp.autocast(True):
        disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)

    # Mark non-overlapping region invalid (matches run_demo.py's remove_invisible)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    invalid = (xx - disp) < 0
    disp[invalid] = np.inf

    with open(K_path) as f:
        K = np.array(list(map(float, f.readline().split()))).reshape(3, 3)
        baseline = float(f.readline())
    depth = K[0, 0] * baseline / np.maximum(disp, 1e-6)
    return depth.astype(np.float32)


def compare_one(capture_dir: Path, model, args):
    print(f"\n=== {capture_dir.name} ===")
    left = capture_dir / "left.png"
    right = capture_dir / "right.png"
    K_path = capture_dir / "K.txt"
    d405_path = capture_dir / "depth_d405.npy"
    color_path = capture_dir / "color.png"
    if not (left.exists() and right.exists() and K_path.exists() and d405_path.exists()):
        print(f"  SKIP — missing files (need left/right/K.txt/depth_d405.npy)")
        return

    fs_depth_path = capture_dir / "fs_depth.npy"
    if fs_depth_path.exists():
        print(f"  reusing cached fs_depth.npy")
        fs_depth = np.load(fs_depth_path)
    else:
        print(f"  running FoundationStereo...")
        fs_depth = run_fs(model, args, left, right, K_path)
        np.save(fs_depth_path, fs_depth)

    d405_depth = np.load(d405_path)
    # depth_d405_measured.npy is depth BEFORE hole_filling — any nonzero pixel
    # here is a genuine measurement, not an extrapolated fill. Use it to gate
    # the error stats. If absent (older captures), fall back to depth_d405.
    measured_path = capture_dir / "depth_d405_measured.npy"
    d405_measured = np.load(measured_path) if measured_path.exists() else d405_depth
    color = cv2.imread(str(color_path)) if color_path.exists() \
            else cv2.imread(str(left))

    # Valid masks
    valid_fs = np.isfinite(fs_depth) & (fs_depth > Z_MIN) & (fs_depth < Z_MAX)
    valid_d4_filled = (d405_depth > Z_MIN) & (d405_depth < Z_MAX)
    valid_d4_meas = (d405_measured > Z_MIN) & (d405_measured < Z_MAX)
    overlap = valid_fs & valid_d4_meas  # fair comparison region

    cov_fs = 100.0 * valid_fs.mean()
    cov_d4 = 100.0 * valid_d4_filled.mean()           # with hole-fill (looks good)
    cov_d4_meas = 100.0 * valid_d4_meas.mean()        # pre hole-fill (truth)
    cov_ov = 100.0 * overlap.mean()
    if overlap.any():
        diff = fs_depth - d405_depth
        diff_ov = diff[overlap]
        rmse = float(np.sqrt((diff_ov ** 2).mean()))
        mean_abs = float(np.abs(diff_ov).mean())
        median_abs = float(np.median(np.abs(diff_ov)))
        bias = float(diff_ov.mean())
    else:
        rmse = mean_abs = median_abs = bias = float("nan")
        diff = np.zeros_like(fs_depth)

    # 2x2 panel: color | D405 | FS | diff
    H, W = fs_depth.shape
    sep_v = np.full((H, 2, 3), 200, dtype=np.uint8)
    sep_h = np.full((2, W * 2 + 2, 3), 200, dtype=np.uint8)
    d405_vis = colorize_depth(d405_depth)
    fs_vis = colorize_depth(fs_depth)
    diff_vis = colorize_diff(np.where(overlap, diff, 0.0))
    top = np.hstack([color, sep_v, d405_vis])
    bot = np.hstack([fs_vis, sep_v, diff_vis])
    grid = np.vstack([top, sep_h, bot])
    for text, pos in [("Color", (10, 25)),
                      (f"D405 ({cov_d4_meas:.0f}% measured / {cov_d4:.0f}% w/hole-fill)", (W + 12, 25)),
                      (f"FoundationStereo ({cov_fs:.0f}% valid)", (10, H + 27)),
                      (f"diff on overlap (RMSE={rmse*1000:.1f}mm med|err|={median_abs*1000:.1f}mm)",
                       (W + 12, H + 27))]:
        cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 3)
        cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 1)

    out_path = capture_dir / "comparison.png"
    cv2.imwrite(str(out_path), grid)

    stats_path = capture_dir / "comparison_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"capture: {capture_dir.name}\n")
        f.write(f"image_size: {W}x{H}\n")
        f.write(f"clip_range_m: [{Z_MIN}, {Z_MAX}]\n\n")
        f.write("Coverage:\n")
        f.write(f"  D405 actually measured : {cov_d4_meas:6.2f} %\n")
        f.write(f"  D405 with hole-filling : {cov_d4:6.2f} %\n")
        f.write(f"  FoundationStereo       : {cov_fs:6.2f} %\n")
        f.write(f"  comparison overlap     : {cov_ov:6.2f} %\n\n")
        f.write("Errors on overlap (FS - D405, D405 = measured only):\n")
        f.write(f"  bias       : {bias*1000:+7.2f} mm   (positive: FS farther)\n")
        f.write(f"  mean |err| : {mean_abs*1000:7.2f} mm\n")
        f.write(f"  median|err|: {median_abs*1000:7.2f} mm\n")
        f.write(f"  RMSE       : {rmse*1000:7.2f} mm\n")
    print(f"  D405 measured: {cov_d4_meas:.1f}%  D405 hole-filled: {cov_d4:.1f}%  "
          f"FS: {cov_fs:.1f}%  overlap: {cov_ov:.1f}%")
    print(f"  bias={bias*1000:+.2f}mm  mean|err|={mean_abs*1000:.2f}mm  "
          f"median|err|={median_abs*1000:.2f}mm  RMSE={rmse*1000:.2f}mm")
    print(f"  wrote {out_path.name} + comparison_stats.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", type=Path,
                        help="capture directories (each must contain left/right/K.txt/depth_d405.npy)")
    parser.add_argument("--latest", action="store_true",
                        help="use the most recent capture under captures/")
    args = parser.parse_args()

    if args.latest:
        root = Path(__file__).parent / "captures"
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not dirs:
            parser.error("no captures found")
        args.dirs = [dirs[-1]]
    if not args.dirs:
        parser.error("pass a capture dir or --latest")

    model, fs_args = load_fs_model()
    for d in args.dirs:
        if not d.is_dir():
            print(f"skip {d} (not a dir)")
            continue
        compare_one(d, model, fs_args)


if __name__ == "__main__":
    main()
