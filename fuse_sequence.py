"""
Fuse a recorded RGB+depth sequence into a single 3D mesh via RGBD odometry +
TSDF volumetric integration.

Pipeline:
  1. Load frames from sequences/<ts>/frames/ — color.png + depth.npy
  2. Frame-to-frame RGBD odometry estimates each frame's pose relative to
     frame 0. Accumulates into a global trajectory.
  3. TSDF volume integrates every K-th frame using its estimated pose.
  4. Extract a triangle mesh + colored point cloud.

Outputs (saved next to frames/):
  trajectory.npy        (N, 4, 4) frame-to-world poses
  fused_mesh.ply        watertight-ish triangle mesh
  fused_cloud.ply       sampled point cloud from the mesh
  fusion_stats.txt      timings, dropped frames, voxel settings

Usage:
  python fuse_sequence.py sequences/<ts>_emON
  python fuse_sequence.py --latest --voxel 0.003 --every 2 --show
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_K(path: Path):
    with open(path) as f:
        K = np.array(list(map(float, f.readline().split()))).reshape(3, 3)
    return K


def make_intrinsic(K, W, H):
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return intr


def load_rgbd(frame_dir: Path, depth_max=1.5):
    color = cv2.imread(str(frame_dir / "color.png"))
    depth_m = np.load(frame_dir / "depth.npy")
    # Open3D expects RGB color and either uint16 (with depth_scale) or float32.
    # Easiest is to build from numpy arrays directly.
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,           # depth is already in meters
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )
    return rgbd, color.shape[:2]


def estimate_trajectory(frames_dir: Path, K, every: int):
    """Frame-to-frame RGBD odometry. Returns list of 4x4 frame-to-world poses
    (length = len(selected_frames)). frame 0 is identity."""
    frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    frame_dirs = frame_dirs[::every]
    print(f"  using {len(frame_dirs)} frames (every {every}-th)")

    rgbd0, (H, W) = load_rgbd(frame_dirs[0])
    intrinsic = make_intrinsic(K, W, H)
    poses = [np.eye(4)]
    odo_opt = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    prev_rgbd = rgbd0
    dropped = 0
    t0 = time.time()
    for i in range(1, len(frame_dirs)):
        curr_rgbd, _ = load_rgbd(frame_dirs[i])
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            curr_rgbd, prev_rgbd, intrinsic, np.eye(4),
            jacobian, odo_opt,
        )
        if success:
            # trans takes curr-frame points into prev-frame coords, so the
            # pose of curr in world = prev_pose @ trans
            poses.append(poses[-1] @ trans)
        else:
            poses.append(poses[-1])
            dropped += 1
        prev_rgbd = curr_rgbd
        if i % 20 == 0:
            print(f"    odometry {i}/{len(frame_dirs)-1}  "
                  f"({(time.time()-t0)/i:.2f}s/frame, {dropped} dropped)")

    print(f"  odometry done in {time.time()-t0:.1f}s ({dropped} dropped)")
    return frame_dirs, poses, intrinsic


def fuse_tsdf(frame_dirs, poses, intrinsic, voxel_size, sdf_trunc, depth_max):
    """Integrate selected frames into a ScalableTSDFVolume."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    t0 = time.time()
    for i, (frame_dir, pose) in enumerate(zip(frame_dirs, poses)):
        rgbd, _ = load_rgbd(frame_dir, depth_max=depth_max)
        # TSDF wants the camera extrinsic (world->camera = inv(pose))
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        if i % 20 == 0:
            print(f"    integrate {i}/{len(frame_dirs)}")
    print(f"  integration done in {time.time()-t0:.1f}s")
    return volume


def process(seq_dir: Path, args):
    print(f"\n=== {seq_dir.name} ===")
    frames_dir = seq_dir / "frames"
    K_path = seq_dir / "K.txt"
    if not frames_dir.is_dir() or not K_path.exists():
        print(f"  SKIP: missing frames/ or K.txt")
        return
    K = load_K(K_path)

    frame_dirs, poses, intrinsic = estimate_trajectory(frames_dir, K, args.every)
    np.save(seq_dir / "trajectory.npy", np.stack(poses))

    volume = fuse_tsdf(frame_dirs, poses, intrinsic,
                       args.voxel, args.sdf_trunc, args.depth_max)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    mesh_path = seq_dir / "fused_mesh.ply"
    cloud_path = seq_dir / "fused_cloud.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(cloud_path), pcd)

    stats_path = seq_dir / "fusion_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"sequence: {seq_dir.name}\n")
        f.write(f"frames_used: {len(frame_dirs)}\n")
        f.write(f"voxel_size_m: {args.voxel}\n")
        f.write(f"sdf_trunc: {args.sdf_trunc}\n")
        f.write(f"depth_max_m: {args.depth_max}\n")
        f.write(f"every_nth_frame: {args.every}\n")
        f.write(f"mesh_vertices: {len(mesh.vertices)}\n")
        f.write(f"mesh_triangles: {len(mesh.triangles)}\n")
        f.write(f"cloud_points: {len(pcd.points)}\n")
    print(f"  wrote {mesh_path.name} ({len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} tris)")
    print(f"  wrote {cloud_path.name} ({len(pcd.points):,} pts)")

    if args.show:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"TSDF fused mesh — {seq_dir.name}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--every", type=int, default=2,
                        help="use every Nth frame (default 2 = halve frame rate)")
    parser.add_argument("--voxel", type=float, default=0.003,
                        help="TSDF voxel size in meters (default 3mm)")
    parser.add_argument("--sdf-trunc", type=float, default=0.012,
                        help="TSDF truncation distance in meters (default 12mm)")
    parser.add_argument("--depth-max", type=float, default=1.5,
                        help="max valid depth in meters")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.latest:
        root = Path(__file__).parent / "sequences"
        if not root.is_dir():
            parser.error("no sequences/ directory yet")
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not dirs:
            parser.error("no sequences recorded")
        args.dirs = [dirs[-1]]
    if not args.dirs:
        parser.error("pass a sequence dir or --latest")

    for d in args.dirs:
        if d.is_dir():
            process(d, args)


if __name__ == "__main__":
    main()
