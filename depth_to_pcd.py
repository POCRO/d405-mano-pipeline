"""
Reproject the saved depth maps into colored point clouds.

For each capture dir created by capture_stereo.py + compare_fs_d405.py, this
script writes:
  cloud_d405.ply    from depth_d405_measured.npy (D405 actual measurements)
  cloud_fs.ply      from fs_depth.npy           (FoundationStereo)

Both clouds live in the left-IR camera frame (same coords), so you can load
them into MeshLab/CloudCompare/Open3D and toggle between them, or pass
--show to pop up an Open3D viewer with both side-by-side.

Usage:
  python depth_to_pcd.py captures/<ts>_emON
  python depth_to_pcd.py --latest --show
  python depth_to_pcd.py captures/* --d405-source filled
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

Z_MIN, Z_MAX = 0.05, 1.5


def load_K(K_path: Path):
    with open(K_path) as f:
        K = np.array(list(map(float, f.readline().split()))).reshape(3, 3)
        baseline = float(f.readline())
    return K, baseline


def depth_to_xyz(depth_m, K):
    H, W = depth_m.shape
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    zs = depth_m
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    return np.stack([xs, ys, zs], axis=-1)


def make_cloud(depth_m, color_bgr, K, z_min=Z_MIN, z_max=Z_MAX):
    """Build a o3d.geometry.PointCloud from a depth map and same-size BGR image."""
    valid = np.isfinite(depth_m) & (depth_m > z_min) & (depth_m < z_max)
    xyz = depth_to_xyz(depth_m, K)
    pts = xyz[valid].reshape(-1, 3)
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)[valid].reshape(-1, 3) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pcd


def process_one(capture_dir: Path, d405_source: str):
    K_path = capture_dir / "K.txt"
    color_path = capture_dir / "color.png"
    fs_path = capture_dir / "fs_depth.npy"
    d405_filled = capture_dir / "depth_d405.npy"
    d405_measured = capture_dir / "depth_d405_measured.npy"

    if not K_path.exists() or not color_path.exists():
        print(f"SKIP {capture_dir.name}: missing K.txt / color.png")
        return None

    K, baseline = load_K(K_path)
    color = cv2.imread(str(color_path))

    out = {}

    if d405_source == "measured" and d405_measured.exists():
        depth = np.load(d405_measured)
        tag = "measured"
    elif d405_filled.exists():
        depth = np.load(d405_filled)
        tag = "filled"
    else:
        depth = None

    if depth is not None:
        pcd = make_cloud(depth, color, K)
        path = capture_dir / "cloud_d405.ply"
        o3d.io.write_point_cloud(str(path), pcd)
        out["d405"] = (pcd, path, len(pcd.points))
        print(f"  cloud_d405.ply  ({tag}, {len(pcd.points):,} pts)")

    if fs_path.exists():
        depth = np.load(fs_path)
        pcd = make_cloud(depth, color, K)
        # Tag FS points with a subtle hue shift so we can tell clouds apart
        # in --show even if they overlap.
        path = capture_dir / "cloud_fs.ply"
        o3d.io.write_point_cloud(str(path), pcd)
        out["fs"] = (pcd, path, len(pcd.points))
        print(f"  cloud_fs.ply    ({len(pcd.points):,} pts)")
    else:
        print(f"  (no fs_depth.npy — run compare_fs_d405.py first to produce FS cloud)")

    return out


def show_interactive(out, initial="both"):
    """Pop one Open3D window with both clouds overlaid in the SAME coordinate
    frame. Use the keyboard to toggle which one is visible:

      D  toggle D405 cloud
      F  toggle FoundationStereo cloud
      1  show only D405
      2  show only FoundationStereo
      3  show both (overlaid)
      Q / ESC  quit

    Overlaying lets you spot per-pixel differences directly — close one and
    open the other to see where holes are, where edges differ, etc.
    """
    pcds = {}
    if "d405" in out:
        pcds["d405"] = out["d405"][0]
    if "fs" in out:
        pcds["fs"] = out["fs"][0]
    if not pcds:
        print("  nothing to show")
        return

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="D405 vs FoundationStereo  [D/F toggle, 1/2/3 view]",
                      width=1280, height=720)
    vis.add_geometry(axis)
    # Track current visibility so we can avoid double-add / double-remove
    # (Open3D segfaults if you remove an already-removed geometry).
    visible = {}
    for name, pcd in pcds.items():
        show_now = (initial in ("both", name))
        if show_now:
            vis.add_geometry(pcd, reset_bounding_box=True)
        visible[name] = show_now

    def set_visible(name, on):
        if name not in pcds or visible[name] == on:
            return
        if on:
            vis.add_geometry(pcds[name], reset_bounding_box=False)
        else:
            vis.remove_geometry(pcds[name], reset_bounding_box=False)
        visible[name] = on

    def toggle(name):
        def _cb(_):
            set_visible(name, not visible[name])
            print(f"  {name}: {'ON' if visible[name] else 'off'}")
            return False
        return _cb

    def only(name):
        def _cb(_):
            for k in pcds:
                set_visible(k, k == name)
            print(f"  showing only {name}")
            return False
        return _cb

    def show_both(_):
        for k in pcds:
            set_visible(k, True)
        print("  showing both (overlaid)")
        return False

    vis.register_key_callback(ord("D"), toggle("d405"))
    vis.register_key_callback(ord("F"), toggle("fs"))
    vis.register_key_callback(ord("1"), only("d405"))
    vis.register_key_callback(ord("2"), only("fs"))
    vis.register_key_callback(ord("3"), show_both)

    print("  viewer keys: D=toggle D405  F=toggle FS  1/2/3=view  Q=quit")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--d405-source", choices=["measured", "filled"],
                        default="measured",
                        help="measured = use depth_d405_measured.npy (no hole fill); "
                             "filled = use depth_d405.npy (hole-filled)")
    parser.add_argument("--show", action="store_true",
                        help="pop up an Open3D window with both clouds overlaid; "
                             "use D/F/1/2/3 keys to toggle which one is visible")
    parser.add_argument("--initial-show", choices=["both", "d405", "fs"],
                        default="both",
                        help="which cloud is visible when the viewer opens "
                             "(can still toggle via keyboard)")
    args = parser.parse_args()

    if args.latest:
        root = Path(__file__).parent / "captures"
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not dirs:
            parser.error("no captures found")
        args.dirs = [dirs[-1]]
    if not args.dirs:
        parser.error("pass a capture dir or --latest")

    for d in args.dirs:
        if not d.is_dir():
            print(f"skip {d}")
            continue
        print(f"\n=== {d.name} ===")
        out = process_one(d, args.d405_source)
        if args.show and out:
            show_interactive(out, initial=args.initial_show)


if __name__ == "__main__":
    main()
