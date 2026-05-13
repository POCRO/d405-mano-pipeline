"""
Run HaMeR on a D405 capture's color.png and dump everything Stage 4 needs.

Outputs in <capture_dir>/hamer_out/:
  mano_hand_<i>.npz   one file per detected hand, contains:
                      vertices_local         (778, 3)  MANO local frame
                      joints_3d_local        ( 21, 3)  MANO local frame
                      vertices_hamer_world   (778, 3)  + cam_t_full applied
                      joints_3d_hamer_world  ( 21, 3)  + cam_t_full applied
                      joints_2d_pixel        ( 21, 2)  pixel coords in input image
                      cam_t_full             ( 3,)     translation in HaMeR cam frame
                      focal_length_hamer_px  ()        HaMeR's virtual focal
                      principal_point_hamer_px (2,)
                      mano_global_orient     (1, 3, 3)
                      mano_hand_pose         (15, 3, 3)
                      mano_betas             (10,)
                      mano_faces             (1538, 3) int32
                      bbox                   (4,)      detected bbox
                      is_right               int       0=left, 1=right
                      image_size             (2,)      (W, H)
                      # Stage 4 (only if depth_d405.npy is present):
                      depth_at_joint_m            (21,)     D405 depth at each joint pixel
                      valid_mask                  (21,) bool
                      joints_3d_d405_perjoint     (21, 3)   each joint ray-marched to D405
                      joints_3d_d405_globalshift  (21, 3)   whole hand shifted by (wrist_d - wrist_h)
                      global_offset_m             ( 3,)
                      d405_K                      ( 3, 3)
  mano_hand_<i>.obj   mesh in HaMeR cam frame (verts + cam_t_full)
  overlay_with_mano.jpg   rendered overlay (same as demo.py's *_all.jpg)

Run inside the .hamer venv:
  source /home/yy/hamer/.hamer/bin/activate
  python infer_hamer.py captures/20260513_150045_emOFF
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HAMER_ROOT = Path("/home/yy/hamer")
sys.path.insert(0, str(HAMER_ROOT))
# HaMeR uses relative paths like _DATA/hamer_ckpts/... so cwd must be its root.
# Save the user's original cwd first so output paths in argv stay correct.
_USER_CWD = Path.cwd()
os.chdir(HAMER_ROOT)

from hamer.configs import CACHE_DIR_HAMER  # noqa: E402
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT  # noqa: E402
from hamer.utils import recursive_to  # noqa: E402
from hamer.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from hamer.utils.renderer import Renderer, cam_crop_to_full  # noqa: E402
from vitpose_model import ViTPoseModel  # noqa: E402

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def load_K(path):
    with open(path) as f:
        K = np.array(list(map(float, f.readline().split()))).reshape(3, 3)
    return K


def detect_hands(img_cv2, detector, cpm):
    """Run vitdet/regnety body detector + ViTPose hand keypoint detector to
    obtain hand bboxes. Returns (bboxes Nx4, is_right N,) or (None, None)."""
    det_out = detector(img_cv2)
    img_rgb = img_cv2.copy()[:, :, ::-1]
    det = det_out['instances']
    valid = (det.pred_classes == 0) & (det.scores > 0.5)
    bboxes = det.pred_boxes.tensor[valid].cpu().numpy()
    scores = det.scores[valid].cpu().numpy()
    if not len(bboxes):
        return None, None
    vit_out = cpm.predict_pose(img_rgb,
                               [np.concatenate([bboxes, scores[:, None]], 1)])
    out_b, out_r = [], []
    for vp in vit_out:
        for side, kpt in [(0, vp['keypoints'][-42:-21]),
                          (1, vp['keypoints'][-21:])]:
            ok = kpt[:, 2] > 0.5
            if ok.sum() > 3:
                bb = [kpt[ok, 0].min(), kpt[ok, 1].min(),
                      kpt[ok, 0].max(), kpt[ok, 1].max()]
                out_b.append(bb)
                out_r.append(side)
    if not out_b:
        return None, None
    return np.stack(out_b), np.array(out_r)


def project_3d(joints_3d, focal, cx, cy):
    u = focal * joints_3d[:, 0] / joints_3d[:, 2] + cx
    v = focal * joints_3d[:, 1] / joints_3d[:, 2] + cy
    return np.stack([u, v], axis=-1)


def sample_depth_at(depth, uv, win=2, z_min=0.05, z_max=1.5):
    """For each (u, v) pixel return the median depth over a (2*win+1) window.
    NaN where no valid sample."""
    H, W = depth.shape
    out = np.full(len(uv), np.nan, dtype=np.float32)
    for i, (u, v) in enumerate(uv):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        patch = depth[max(0, vi - win):vi + win + 1,
                      max(0, ui - win):ui + win + 1]
        valid = patch[(patch > z_min) & (patch < z_max)]
        if valid.size:
            out[i] = float(np.median(valid))
    return out


def refine_with_depth(joints_3d_h, joints_2d_px, depth_path, K_d):
    """Two refinement variants:
    - per_joint: each joint Z replaced with D405 depth at its 2D pixel,
                 then back-projected through D405 intrinsics. Breaks bone
                 lengths but exact-fits the depth surface.
    - globalshift: shift the whole hand by (wrist_d405 - wrist_hamer).
                 Preserves bone lengths; fixes only the global Z bias.
    """
    depth = np.load(depth_path)
    fx, fy, cx, cy = K_d[0, 0], K_d[1, 1], K_d[0, 2], K_d[1, 2]

    z_d = sample_depth_at(depth, joints_2d_px)
    valid = ~np.isnan(z_d)

    per_joint = joints_3d_h.copy()
    for i in range(len(joints_3d_h)):
        if valid[i]:
            u, v = joints_2d_px[i]
            z = z_d[i]
            per_joint[i] = np.array([(u - cx) * z / fx,
                                     (v - cy) * z / fy, z])

    if valid[0]:
        u_w, v_w = joints_2d_px[0]
        z_w = z_d[0]
        wrist_d = np.array([(u_w - cx) * z_w / fx,
                            (v_w - cy) * z_w / fy, z_w])
        offset = wrist_d - joints_3d_h[0]
        global_shift = joints_3d_h + offset
    else:
        global_shift = None
        offset = None

    return {
        'depth_at_joint_m': z_d,
        'valid_mask': valid,
        'joints_3d_d405_perjoint': per_joint.astype(np.float32),
        'joints_3d_d405_globalshift': (global_shift.astype(np.float32)
                                       if global_shift is not None else np.array([])),
        'global_offset_m': (offset.astype(np.float32)
                            if offset is not None else np.array([])),
        'd405_K': K_d.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture_dir', type=Path,
                        help='captures/<ts>_em*/  (must contain color.png + K.txt)')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--detector', choices=['vitdet', 'regnety'],
                        default='regnety',
                        help='regnety fits in 8GB; vitdet needs 12GB+ but is more accurate')
    parser.add_argument('--no-depth-refine', action='store_true')
    args = parser.parse_args()

    # capture_dir was given relative to the user's original cwd, not HaMeR's
    cap_dir = args.capture_dir
    if not cap_dir.is_absolute():
        cap_dir = (_USER_CWD / cap_dir).resolve()
    color_path = cap_dir / 'color.png'
    K_path = cap_dir / 'K.txt'
    depth_path = cap_dir / 'depth_d405.npy'
    if not color_path.exists() or not K_path.exists():
        sys.exit(f"missing color.png or K.txt in {cap_dir}")
    K_d = load_K(K_path)

    out_dir = cap_dir / 'hamer_out'
    out_dir.mkdir(exist_ok=True)

    # --- HaMeR setup ---
    # Skip download_models() — we already have everything in _DATA/.
    # If you ever delete _DATA/, restore by running:
    #   cd /home/yy/hamer && python -c "from hamer.models import download_models, CACHE_DIR_HAMER; download_models(CACHE_DIR_HAMER)"
    if not (HAMER_ROOT / '_DATA' / 'hamer_ckpts' / 'model_config.yaml').exists():
        sys.exit("missing _DATA/hamer_ckpts — please re-run download_models manually")
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'regnety':
        from detectron2 import model_zoo
        cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
            trained=True)
        cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(cfg)
    else:
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        cfg = LazyConfig.load(str(cfg_path))
        cfg.train.init_checkpoint = ("https://dl.fbaipublicfiles.com/detectron2/"
                                     "ViTDet/COCO/cascade_mask_rcnn_vitdet_h/"
                                     "f328730692/model_final_f05665.pkl")
        for i in range(3):
            cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(cfg)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    img_cv2 = cv2.imread(str(color_path))
    H_img, W_img = img_cv2.shape[:2]

    boxes, is_right_arr = detect_hands(img_cv2, detector, cpm)
    if boxes is None:
        sys.exit("no hands detected in color.png")

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, is_right_arr,
                            rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=0)

    all_verts_local = []   # MANO local (mirrored) — for renderer
    all_cam_t_full = []
    all_right = []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam'].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch['box_center'].float()
        box_size = batch['box_size'].float()
        img_size_t = batch['img_size'].float()
        scaled_focal_length = (model_cfg.EXTRA.FOCAL_LENGTH /
                               model_cfg.MODEL.IMAGE_SIZE *
                               img_size_t.max())
        focal_h = float(scaled_focal_length.detach().cpu().item())
        cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size,
                                      img_size_t, scaled_focal_length
                                      ).detach().cpu().numpy()

        B = batch['img'].shape[0]
        for n in range(B):
            person_id = int(batch['personid'][n])
            right_n = int(batch['right'][n].item())

            verts_local = out['pred_vertices'][n].detach().cpu().numpy()
            joints_local = out['pred_keypoints_3d'][n].detach().cpu().numpy()

            # Mirror left hand X (HaMeR convention from demo.py)
            verts_local_m = verts_local.copy()
            joints_local_m = joints_local.copy()
            verts_local_m[:, 0] = (2 * right_n - 1) * verts_local_m[:, 0]
            joints_local_m[:, 0] = (2 * right_n - 1) * joints_local_m[:, 0]

            t_full = cam_t_full[n]
            verts_world = verts_local_m + t_full
            joints_world = joints_local_m + t_full

            cx_h, cy_h = W_img / 2.0, H_img / 2.0
            joints_2d_px = project_3d(joints_world, focal_h, cx_h, cy_h)

            mp = out['pred_mano_params']
            payload = dict(
                vertices_local=verts_local_m.astype(np.float32),
                joints_3d_local=joints_local_m.astype(np.float32),
                vertices_hamer_world=verts_world.astype(np.float32),
                joints_3d_hamer_world=joints_world.astype(np.float32),
                joints_2d_pixel=joints_2d_px.astype(np.float32),
                cam_t_full=t_full.astype(np.float32),
                focal_length_hamer_px=np.float32(focal_h),
                principal_point_hamer_px=np.array([cx_h, cy_h], dtype=np.float32),
                mano_global_orient=mp['global_orient'][n].detach().cpu().numpy().astype(np.float32),
                mano_hand_pose=mp['hand_pose'][n].detach().cpu().numpy().astype(np.float32),
                mano_betas=mp['betas'][n].detach().cpu().numpy().astype(np.float32),
                mano_faces=np.asarray(model.mano.faces, dtype=np.int32),
                bbox=boxes[person_id].astype(np.float32),
                is_right=np.int32(right_n),
                image_size=np.array([W_img, H_img], dtype=np.int32),
            )

            if depth_path.exists() and not args.no_depth_refine:
                payload.update(refine_with_depth(joints_world, joints_2d_px,
                                                 depth_path, K_d))

            np.savez(out_dir / f'mano_hand_{person_id}.npz', **payload)

            import trimesh
            mesh = trimesh.Trimesh(vertices=verts_world,
                                   faces=model.mano.faces, process=False)
            mesh.export(out_dir / f'mano_hand_{person_id}.obj')

            all_verts_local.append(verts_local_m)
            all_cam_t_full.append(t_full)
            all_right.append(right_n)

            msg = (f"[hand {person_id}] right={right_n}  "
                   f"wrist_z_hamer={joints_world[0, 2]:.3f}m  ")
            if 'global_offset_m' in payload and payload['global_offset_m'].size:
                msg += (f"wrist_z_d405={payload['joints_3d_d405_globalshift'][0, 2]:.3f}m  "
                        f"global_dz={payload['global_offset_m'][2] * 1000:+.1f}mm  "
                        f"valid_joints={int(payload['valid_mask'].sum())}/21")
            print(msg)

    # Full-frame overlay (same as demo.py *_all.jpg)
    img_size_overlay = torch.tensor([W_img, H_img], dtype=torch.float32, device=device)
    sfl = (model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE *
           img_size_overlay.max())
    cam_view = renderer.render_rgba_multiple(
        all_verts_local, cam_t=all_cam_t_full, render_res=img_size_overlay,
        is_right=all_right,
        mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
        focal_length=sfl)
    base = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    base = np.concatenate([base, np.ones_like(base[:, :, :1])], axis=2)
    overlay = (base[:, :, :3] * (1 - cam_view[:, :, 3:]) +
               cam_view[:, :, :3] * cam_view[:, :, 3:])
    cv2.imwrite(str(out_dir / 'overlay_with_mano.jpg'),
                255 * overlay[:, :, ::-1])

    print(f"\nWrote {len(all_verts_local)} hand(s) -> {out_dir}/")


if __name__ == '__main__':
    main()
