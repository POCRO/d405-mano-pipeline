"""
Fit MANO parameters to D405 depth observations.

Stage 4 (the basic version in infer_hamer.py) only shifts the wrist to match
D405 depth, treating the whole hand as a rigid body. This script does the
*real* depth refinement: it optimizes MANO's (β, θ, global_orient, T) so that
the predicted joints, when projected through D405's actual intrinsics, hit
D405's depth surface at every joint — using all 21 depth observations rather
than just the wrist.

Inputs (in <capture_dir>/):
  hamer_out/mano_hand_<i>.npz   — produced by infer_hamer.py
  depth_d405.npy                — D405 metric depth in meters
  K.txt                         — D405 intrinsics

Outputs:
  hamer_out/mano_hand_<i>_fitted.npz   refined params + joints + vertices
  hamer_out/mano_hand_<i>_fitted.obj   mesh for inspection

Run inside the .hamer venv:
  source /home/yy/hamer/.hamer/bin/activate
  python fit_mano.py captures/20260513_150045_emOFF
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HAMER_ROOT = Path("/home/yy/hamer")
sys.path.insert(0, str(HAMER_ROOT))
_USER_CWD = Path.cwd()
os.chdir(HAMER_ROOT)

from hamer.models import load_hamer, DEFAULT_CHECKPOINT  # noqa: E402


# --------------------------------------------------------------------------
# Axis-angle ↔ rotation matrix (Rodrigues). We optimize in axis-angle because
# it's 3 unconstrained numbers per rotation. HaMeR stores rotmat, so we
# convert at the boundary.
# --------------------------------------------------------------------------
def axis_angle_to_matrix(aa):
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    a = angle.unsqueeze(-1)
    return I + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)


def matrix_to_axis_angle(R):
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos)
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    sin = torch.sin(angle).clamp(min=1e-6)
    axis = axis / (2 * sin.unsqueeze(-1))
    return axis * angle.unsqueeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture_dir', type=Path)
    ap.add_argument('--hand', type=int, default=0, help='which detected hand (0/1)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--w-depth', type=float, default=1000.0,
                    help='weight for depth residual (loss in m², so 1000 ~ 30mm RMS scale)')
    ap.add_argument('--w-2d', type=float, default=0.0001,
                    help='weight for keeping projected 2D near HaMeR-predicted 2D')
    ap.add_argument('--w-betas', type=float, default=10.0,
                    help='penalize β drifting from HaMeR init')
    ap.add_argument('--w-pose', type=float, default=10.0,
                    help='penalize θ drifting from HaMeR init')
    ap.add_argument('--fix-betas', action='store_true',
                    help='do not optimize β (recommended for single-frame fits)')
    args = ap.parse_args()

    cap_dir = args.capture_dir
    if not cap_dir.is_absolute():
        cap_dir = (_USER_CWD / cap_dir).resolve()
    npz_path = cap_dir / 'hamer_out' / f'mano_hand_{args.hand}.npz'
    depth_path = cap_dir / 'depth_d405.npy'
    if not npz_path.exists() or not depth_path.exists():
        sys.exit(f"need {npz_path.name} and depth_d405.npy in {cap_dir}")

    npz = dict(np.load(npz_path))
    depth_np = np.load(depth_path)
    H_img, W_img = depth_np.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not (HAMER_ROOT / '_DATA' / 'hamer_ckpts' / 'model_config.yaml').exists():
        sys.exit("missing HaMeR data — see infer_hamer.py docstring")
    model, _ = load_hamer(DEFAULT_CHECKPOINT)
    mano = model.mano.to(device).eval()
    for p in mano.parameters():
        p.requires_grad_(False)

    is_right = int(npz['is_right'])

    # --- HaMeR-initialized params (used as init AND as prior) ---
    init_orient_R = torch.tensor(npz['mano_global_orient']).to(device)  # (1, 3, 3)
    init_pose_R = torch.tensor(npz['mano_hand_pose']).to(device)        # (15, 3, 3)
    init_betas = torch.tensor(npz['mano_betas']).to(device)             # (10,)
    init_orient_aa = matrix_to_axis_angle(init_orient_R)                # (1, 3)
    init_pose_aa = matrix_to_axis_angle(init_pose_R)                    # (15, 3)
    mirror = torch.tensor([-1.0, 1.0, 1.0] if is_right == 0 else [1.0, 1.0, 1.0],
                          device=device)

    # Initial wrist position: use the Stage 4 global-shift result as a warm start.
    target_wrist = torch.tensor(npz['joints_3d_d405_globalshift'][0]).to(device)
    # Run MANO once to figure out the local-frame wrist offset, then init T so
    # the MANO output's joints[0] starts at target_wrist.
    with torch.no_grad():
        out_init = mano(
            betas=init_betas[None],
            global_orient=init_orient_R[None],
            hand_pose=init_pose_R[None],
        )
        wrist_local_init = out_init.joints[0, 0] * mirror   # (3,)
    init_T = target_wrist - wrist_local_init

    # --- Optimization variables ---
    aa_pose = init_pose_aa.detach().clone().requires_grad_(True)
    aa_orient = init_orient_aa.detach().clone().requires_grad_(True)
    betas = init_betas.detach().clone()
    betas.requires_grad_(not args.fix_betas)
    T = init_T.detach().clone().requires_grad_(True)

    init_pose_aa_c = init_pose_aa.detach().clone()
    init_orient_aa_c = init_orient_aa.detach().clone()
    init_betas_c = init_betas.detach().clone()

    K_d = torch.tensor(npz['d405_K']).to(device)
    fx, fy, cx, cy = K_d[0, 0], K_d[1, 1], K_d[0, 2], K_d[1, 2]
    target_2d = torch.tensor(npz['joints_2d_pixel']).to(device)
    depth_t = torch.tensor(depth_np, dtype=torch.float32, device=device)

    params = [aa_pose, aa_orient, T]
    if not args.fix_betas:
        params.append(betas)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    def forward():
        R_o = axis_angle_to_matrix(aa_orient)        # (1, 3, 3)
        R_p = axis_angle_to_matrix(aa_pose)          # (15, 3, 3)
        out = mano(betas=betas[None],
                   global_orient=R_o[None],
                   hand_pose=R_p[None])
        joints_local = out.joints[0] * mirror        # (21, 3)
        verts_local = out.vertices[0] * mirror       # (778, 3)
        return joints_local + T, verts_local + T

    def project(p3d):
        u = fx * p3d[:, 0] / p3d[:, 2] + cx
        v = fy * p3d[:, 1] / p3d[:, 2] + cy
        return torch.stack([u, v], dim=-1)

    def sample_depth(uv):
        # grid_sample wants normalized [-1, 1] (x, y) coords with align_corners=True
        grid_x = uv[:, 0] / (W_img - 1) * 2 - 1
        grid_y = uv[:, 1] / (H_img - 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)[None, None]
        z = F.grid_sample(depth_t[None, None], grid, mode='bilinear',
                          padding_mode='zeros', align_corners=True).reshape(-1)
        return z

    print(f"=== fit_mano.py | {cap_dir.name} | hand {args.hand} ===")
    print(f"init wrist target (Stage 4): {target_wrist.cpu().numpy()}")
    print(f"steps={args.steps}  lr={args.lr}  fix_betas={args.fix_betas}")
    print(f"weights: depth={args.w_depth} 2d={args.w_2d} "
          f"betas={args.w_betas} pose={args.w_pose}\n")

    for step in range(args.steps):
        optimizer.zero_grad()
        joints_w, verts_w = forward()
        uv = project(joints_w)
        z_sampled = sample_depth(uv)
        valid = (z_sampled > 0.05) & (z_sampled < 1.5)

        if valid.sum() == 0:
            print(f"step {step}: no valid depth — projection outside image?")
            break

        depth_resid = (joints_w[:, 2] - z_sampled)[valid]
        loss_depth = (depth_resid ** 2).mean()
        loss_2d = ((uv - target_2d) ** 2).sum(dim=-1).mean()
        loss_betas = ((betas - init_betas_c) ** 2).mean()
        loss_pose = (((aa_pose - init_pose_aa_c) ** 2).mean() +
                     ((aa_orient - init_orient_aa_c) ** 2).mean())
        loss = (args.w_depth * loss_depth +
                args.w_2d * loss_2d +
                args.w_betas * loss_betas +
                args.w_pose * loss_pose)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                rms_d = (depth_resid ** 2).mean().sqrt() * 1000
                rms_2d = ((uv - target_2d) ** 2).sum(dim=-1).mean().sqrt()
                wz = joints_w[0, 2].item()
            print(f"  step {step:3d}  L={loss.item():9.3f}  "
                  f"d_rms={rms_d:6.2f}mm  2d_rms={rms_2d:5.2f}px  "
                  f"wrist_z={wz*100:5.2f}cm  valid={int(valid.sum())}/21")

    # --- Save ---
    with torch.no_grad():
        joints_w, verts_w = forward()
        R_o = axis_angle_to_matrix(aa_orient).cpu().numpy()
        R_p = axis_angle_to_matrix(aa_pose).cpu().numpy()
        out_path = cap_dir / 'hamer_out' / f'mano_hand_{args.hand}_fitted.npz'
        np.savez(out_path,
                 vertices_fitted=verts_w.cpu().numpy().astype(np.float32),
                 joints_fitted=joints_w.cpu().numpy().astype(np.float32),
                 mano_betas_fitted=betas.cpu().numpy().astype(np.float32),
                 mano_hand_pose_fitted=R_p.astype(np.float32),
                 mano_global_orient_fitted=R_o.astype(np.float32),
                 transl_fitted=T.cpu().numpy().astype(np.float32),
                 is_right=np.int32(is_right),
                 d405_K=npz['d405_K'])
        print(f"\nWrote {out_path.name}")

        import trimesh
        mesh = trimesh.Trimesh(vertices=verts_w.cpu().numpy(),
                               faces=npz['mano_faces'], process=False)
        mesh.export(cap_dir / 'hamer_out' / f'mano_hand_{args.hand}_fitted.obj')

        # Quick side-by-side: HaMeR-only joints vs fitted joints, on D405 depth
        hamer_joints = npz['joints_3d_d405_globalshift']
        bone_pairs = [(0, 1), (1, 2), (2, 3), (3, 4),     # thumb
                      (0, 5), (5, 6), (6, 7), (7, 8),     # index
                      (0, 9), (9, 10), (10, 11), (11, 12),# middle
                      (0, 13), (13, 14), (14, 15), (15, 16), # ring
                      (0, 17), (17, 18), (18, 19), (19, 20)] # pinky
        print("\n=== bone lengths (mm) — HaMeR-rigid-shift vs depth-fitted ===")
        print(f"{'bone':10s} {'HaMeR':>8s} {'fitted':>8s} {'Δ':>6s}")
        for (a, b) in bone_pairs:
            la = np.linalg.norm(hamer_joints[a] - hamer_joints[b]) * 1000
            jf = joints_w.cpu().numpy()
            lb = np.linalg.norm(jf[a] - jf[b]) * 1000
            print(f"{a:>2d}-{b:<2d}      {la:8.2f} {lb:8.2f} {lb-la:+6.2f}")


if __name__ == '__main__':
    main()
