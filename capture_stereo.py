"""
Capture rectified stereo IR pairs + D405 native depth + color for FoundationStereo
comparison.

D405's two IR streams (stream index 1 = left, 2 = right) come out of the SDK
already rectified — same Y row corresponds across the pair, no distortion.
That is what FoundationStereo expects. D405's native depth is computed in the
same coordinate frame as IR1, so the FS depth and D405 depth can be compared
pixel-for-pixel without alignment.

Controls:
  SPACE  save everything into captures/<timestamp>/
  E      toggle the IR projector (emitter) on/off
  Q      quit

Each captured timestamp directory contains:
  left.png         IR left as BGR (replicated grayscale, what run_demo.py wants)
  right.png        IR right as BGR
  color.png        RGB color stream
  depth_d405.npy   D405 native depth in meters, after spatial+temporal+hole_filling
  depth_d405_raw.png  raw Z16 depth (mm), 16-bit png, no filters
  depth_d405_vis.png  colorized depth for quick eyeballing
  K.txt            FoundationStereo intrinsics+baseline (left IR's frame)
  meta.txt         capture metadata (emitter state, depth_scale, timestamps)
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

W, H, FPS = 848, 480, 30
CAPTURE_ROOT = Path(__file__).parent / "captures"
CAPTURE_ROOT.mkdir(exist_ok=True)

# Comparison range — D405's useful working volume is ~7-50cm. Outside that,
# both methods get unreliable, and clipping makes the colormaps comparable.
Z_MIN, Z_MAX = 0.05, 1.5


def build_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    profile = pipeline.start(config)
    return pipeline, profile


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


def write_k_txt(path, K, baseline):
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in K.flatten()) + "\n")
        f.write(f"{baseline:.6f}\n")


def make_filter_chain():
    """Return (pre_hf_filters, hole_filling_filter).

    Decimation is skipped because it changes resolution and breaks pixel-aligned
    comparison with FoundationStereo. Hole filling is split out so we can save
    both pre- and post-hole-filling depth — the comparison script uses the
    pre-hole-filling version as the "actually measured" mask, otherwise D405's
    extrapolated holes look like real measurements and inflate the error stats.
    """
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, Z_MIN)
    threshold.set_option(rs.option.max_distance, Z_MAX)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    return [threshold, spatial, temporal], hole_filling


def apply_filters(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


def colorize_depth(depth_m, z_min=Z_MIN, z_max=Z_MAX):
    """Float meters -> 8-bit BGR using TURBO colormap. Invalid -> black."""
    valid = (depth_m > z_min) & (depth_m < z_max)
    norm = np.zeros_like(depth_m, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, z_min, z_max)
        norm_f = (clipped - z_min) / (z_max - z_min) * 255.0
        norm = norm_f.astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def main():
    pipeline, profile = build_pipeline()
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per Z16 unit, ~0.0001 for D405
    if depth_sensor.supports(rs.option.emitter_enabled):
        # Start with emitter ON — that's D405's intended operating mode and
        # D405's native SGBM needs it for textureless surfaces. Toggle with E.
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        emitter_on = True
    else:
        emitter_on = False

    K, baseline, intr = read_intrinsics(profile)
    align = rs.align(rs.stream.infrared)  # align color+depth to IR1's frame
    pre_hf_filters, hole_filling = make_filter_chain()

    print(f"D405 IR intrinsics: fx={intr.fx:.3f} fy={intr.fy:.3f} "
          f"ppx={intr.ppx:.3f} ppy={intr.ppy:.3f}")
    print(f"D405 IR baseline:   {baseline*1000:.3f} mm")
    print(f"depth_scale:        {depth_scale:.6f} m/unit")
    print(f"Captures will go to: {CAPTURE_ROOT}")
    print("[SPACE] save  [E] toggle emitter  [Q] quit")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            ir_l = aligned.get_infrared_frame(1)
            ir_r = aligned.get_infrared_frame(2)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not ir_l or not ir_r or not depth_f or not color_f:
                continue

            left = np.asanyarray(ir_l.get_data())
            right = np.asanyarray(ir_r.get_data())
            color_img = np.asanyarray(color_f.get_data())
            depth_pre_hf = apply_filters(depth_f, pre_hf_filters)
            depth_post_hf = hole_filling.process(depth_pre_hf)
            depth_m_measured = np.asanyarray(depth_pre_hf.get_data()).astype(np.float32) * depth_scale
            depth_m = np.asanyarray(depth_post_hf.get_data()).astype(np.float32) * depth_scale

            # 2x2 preview: IR-L | IR-R / Color | D405 depth
            sep_v = np.full((H, 2, 3), 200, dtype=np.uint8)
            sep_h = np.full((2, W * 2 + 2, 3), 200, dtype=np.uint8)
            ir_l_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            ir_r_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            depth_vis = colorize_depth(depth_m)
            top = np.hstack([ir_l_bgr, sep_v, ir_r_bgr])
            bot = np.hstack([color_img, sep_v, depth_vis])
            grid = np.vstack([top, sep_h, bot])
            for text, pos in [("IR left", (10, 25)),
                              ("IR right", (W + 12, 25)),
                              ("Color", (10, H + 27)),
                              ("D405 depth (filt.)", (W + 12, H + 27))]:
                cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            status = (f"emitter={'ON' if emitter_on else 'OFF'}  "
                      "SPACE=save  E=emitter  Q=quit")
            cv2.putText(grid, status, (10, 2 * H + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if emitter_on else (180, 180, 180), 2)

            cv2.imshow("D405 capture  [SPACE] save  [E] emitter  [Q] quit", grid)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('e') and depth_sensor.supports(rs.option.emitter_enabled):
                emitter_on = not emitter_on
                depth_sensor.set_option(rs.option.emitter_enabled,
                                        1 if emitter_on else 0)
                print(f"emitter -> {'ON' if emitter_on else 'OFF'}")
            if key == ord(' '):
                ts = time.strftime("%Y%m%d_%H%M%S")
                tag = "emON" if emitter_on else "emOFF"
                out_dir = CAPTURE_ROOT / f"{ts}_{tag}"
                out_dir.mkdir(exist_ok=True)

                cv2.imwrite(str(out_dir / "left.png"),  ir_l_bgr)
                cv2.imwrite(str(out_dir / "right.png"), ir_r_bgr)
                cv2.imwrite(str(out_dir / "color.png"), color_img)
                # Raw Z16 (no filters), 16-bit PNG — exact original depth
                raw_z16 = np.asanyarray(depth_f.get_data())
                cv2.imwrite(str(out_dir / "depth_d405_raw.png"), raw_z16)
                # Filtered metric depth as float32. _measured is what the
                # sensor actually returned (after spatial+temporal, before hole
                # filling) — compare_fs_d405.py uses non-zero entries here as
                # the mask of "where D405 actually had a measurement". depth_m
                # is the post-hole-filling version for visualization only.
                np.save(out_dir / "depth_d405.npy", depth_m)
                np.save(out_dir / "depth_d405_measured.npy", depth_m_measured)
                cv2.imwrite(str(out_dir / "depth_d405_vis.png"), depth_vis)
                write_k_txt(out_dir / "K.txt", K, baseline)
                with open(out_dir / "meta.txt", "w") as f:
                    f.write(f"timestamp: {ts}\n")
                    f.write(f"emitter_on: {emitter_on}\n")
                    f.write(f"depth_scale_m_per_unit: {depth_scale}\n")
                    f.write(f"resolution: {W}x{H}\n")
                    f.write(f"fps: {FPS}\n")
                    f.write(f"z_clip_range_m: [{Z_MIN}, {Z_MAX}]\n")
                    f.write("filters: threshold + spatial + temporal + hole_filling\n")
                print(f"saved -> {out_dir}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
