"""
Record a continuous sequence from D405 — RGB + filtered depth + stereo IR pair
per frame — for downstream multi-view 3D fusion.

Workflow: press SPACE to start recording, press SPACE again to stop.
Each frame is saved as:
  sequences/<ts>/
    frames/000000/{color.png, depth.png, depth.npy, left.png, right.png}
    frames/000001/...
    K.txt          intrinsics (same as capture_stereo.py)
    meta.txt       fps, depth_scale, emitter state, frame count

depth.png is the post-hole-fill 16-bit Z16, depth.npy is the float32 metric
depth (already in left-IR frame). All frames share the same intrinsics — the
fusion script (fuse_sequence.py) consumes this layout directly.

Controls:
  SPACE  start / stop recording (toggle)
  E      toggle IR projector (kept ON by default — D405's intended mode)
  Q      quit
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

W, H, FPS = 848, 480, 30
SEQUENCE_ROOT = Path(__file__).parent / "sequences"
SEQUENCE_ROOT.mkdir(exist_ok=True)

Z_MIN, Z_MAX = 0.05, 1.5


def build_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    return pipeline, pipeline.start(config)


def read_intrinsics(profile):
    ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    intr = ir1.get_intrinsics()
    extr = ir1.get_extrinsics_to(ir2)
    baseline = float(np.linalg.norm(extr.translation))
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1.0]], dtype=np.float64)
    return K, baseline, intr


def make_filters():
    th = rs.threshold_filter()
    th.set_option(rs.option.min_distance, Z_MIN)
    th.set_option(rs.option.max_distance, Z_MAX)
    sp = rs.spatial_filter()
    sp.set_option(rs.option.filter_magnitude, 2)
    sp.set_option(rs.option.filter_smooth_alpha, 0.5)
    sp.set_option(rs.option.filter_smooth_delta, 20)
    tp = rs.temporal_filter()
    tp.set_option(rs.option.filter_smooth_alpha, 0.4)
    tp.set_option(rs.option.filter_smooth_delta, 20)
    hf = rs.hole_filling_filter()
    hf.set_option(rs.option.holes_fill, 1)
    return [th, sp, tp, hf]


def colorize_depth(depth_m):
    valid = (depth_m > Z_MIN) & (depth_m < Z_MAX)
    norm = np.zeros_like(depth_m, dtype=np.uint8)
    if valid.any():
        norm = ((np.clip(depth_m, Z_MIN, Z_MAX) - Z_MIN) /
                (Z_MAX - Z_MIN) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def main():
    pipeline, profile = build_pipeline()
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        emitter_on = True
    else:
        emitter_on = False

    K, baseline, intr = read_intrinsics(profile)
    align = rs.align(rs.stream.infrared)
    filters = make_filters()

    print(f"D405 intrinsics fx={intr.fx:.3f} ppx={intr.ppx:.3f} "
          f"ppy={intr.ppy:.3f}  baseline={baseline*1000:.2f}mm")
    print("[SPACE] start/stop recording  [E] emitter  [Q] quit")

    recording = False
    seq_dir = None
    frames_dir = None
    frame_idx = 0
    start_time = None

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            ir_l = aligned.get_infrared_frame(1)
            ir_r = aligned.get_infrared_frame(2)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not (ir_l and ir_r and color_f and depth_f):
                continue

            left = np.asanyarray(ir_l.get_data())
            right = np.asanyarray(ir_r.get_data())
            color_img = np.asanyarray(color_f.get_data())
            depth_filtered = depth_f
            for f in filters:
                depth_filtered = f.process(depth_filtered)
            depth_m = (np.asanyarray(depth_filtered.get_data()).astype(np.float32)
                       * depth_scale)
            depth_z16 = np.asanyarray(depth_filtered.get_data()).astype(np.uint16)

            if recording:
                frame_dir = frames_dir / f"{frame_idx:06d}"
                frame_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(frame_dir / "color.png"), color_img)
                cv2.imwrite(str(frame_dir / "depth.png"), depth_z16)
                np.save(frame_dir / "depth.npy", depth_m)
                cv2.imwrite(str(frame_dir / "left.png"),
                            cv2.cvtColor(left, cv2.COLOR_GRAY2BGR))
                cv2.imwrite(str(frame_dir / "right.png"),
                            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
                frame_idx += 1

            # 2x2 preview: color | depth_vis / IR_left | IR_right
            depth_vis = colorize_depth(depth_m)
            ir_l_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            ir_r_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            sep_v = np.full((H, 2, 3), 200, dtype=np.uint8)
            sep_h = np.full((2, W * 2 + 2, 3), 200, dtype=np.uint8)
            top = np.hstack([color_img, sep_v, depth_vis])
            bot = np.hstack([ir_l_bgr, sep_v, ir_r_bgr])
            grid = np.vstack([top, sep_h, bot])

            elapsed = time.time() - start_time if start_time else 0.0
            status = (f"REC #{frame_idx}  {elapsed:5.1f}s" if recording
                      else "stopped")
            color = (0, 0, 255) if recording else (180, 180, 180)
            cv2.putText(grid, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(grid, f"emitter={'ON' if emitter_on else 'OFF'}  "
                              "SPACE=rec  E=emitter  Q=quit",
                        (10, 2 * H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cv2.imshow("D405 sequence recorder", grid)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('e') and depth_sensor.supports(rs.option.emitter_enabled):
                emitter_on = not emitter_on
                depth_sensor.set_option(rs.option.emitter_enabled,
                                        1 if emitter_on else 0)
            if key == ord(' '):
                if not recording:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    tag = "emON" if emitter_on else "emOFF"
                    seq_dir = SEQUENCE_ROOT / f"{ts}_{tag}"
                    seq_dir.mkdir(exist_ok=True)
                    frames_dir = seq_dir / "frames"
                    frames_dir.mkdir(exist_ok=True)
                    with open(seq_dir / "K.txt", "w") as f:
                        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
                        f.write(f"{baseline:.6f}\n")
                    start_time = time.time()
                    frame_idx = 0
                    recording = True
                    print(f"REC START -> {seq_dir}")
                else:
                    recording = False
                    with open(seq_dir / "meta.txt", "w") as f:
                        f.write(f"frame_count: {frame_idx}\n")
                        f.write(f"fps_target: {FPS}\n")
                        f.write(f"duration_s: {time.time()-start_time:.2f}\n")
                        f.write(f"depth_scale_m_per_unit: {depth_scale}\n")
                        f.write(f"emitter_on: {emitter_on}\n")
                        f.write(f"resolution: {W}x{H}\n")
                        f.write(f"z_clip_range_m: [{Z_MIN}, {Z_MAX}]\n")
                    print(f"REC STOP  ({frame_idx} frames, "
                          f"{time.time()-start_time:.1f}s) -> {seq_dir}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
