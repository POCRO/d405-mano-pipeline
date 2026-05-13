import numpy as np
import cv2
import pyrealsense2 as rs

W, H, FPS = 640, 480, 15

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth,    W, H, rs.format.z16,  FPS)
config.enable_stream(rs.stream.color,    W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.infrared, W, H, rs.format.y8,   FPS)

profile = pipeline.start(config)

# align depth to color frame
align = rs.align(rs.stream.color)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 2)  # 2 = WhiteToBlack

# post-processing filter chain
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 2)  # 降采样倍数

threshold = rs.threshold_filter()
threshold.set_option(rs.option.min_distance, 0.1)   # 最近 0.1m
threshold.set_option(rs.option.max_distance, 1.5)   # 最远 1.5m（D405 近距离相机）

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

hole_filling = rs.hole_filling_filter()
hole_filling.set_option(rs.option.holes_fill, 1)  # 1 = farthest from around

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale:.4f} m/unit  |  Press Q to quit")

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        ir_frame    = frames.get_infrared_frame()

        if not depth_frame or not color_frame or not ir_frame:
            continue

        # apply post-processing filter chain
        depth_frame = decimation.process(depth_frame)
        depth_frame = threshold.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_img = cv2.resize(depth_img, (W, H))
        ir_img    = np.asanyarray(ir_frame.get_data())
        ir_bgr    = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)

        # overlay depth value at center pixel (decimation shrinks the frame)
        depth_vf = depth_frame.as_video_frame()
        dw, dh = depth_vf.width, depth_vf.height
        cx, cy = dw // 2, dh // 2
        dist_m = depth_frame.as_depth_frame().get_distance(cx, cy)
        # draw marker at color-space center on both panels
        mcx, mcy = W // 2, H // 2
        for img in (color_img, depth_img):
            cv2.drawMarker(img, (mcx, mcy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, f"{dist_m:.3f} m", (mcx + 12, mcy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        top    = np.hstack([color_img, depth_img])
        bottom = np.hstack([ir_bgr, np.zeros_like(ir_bgr)])  # placeholder
        grid   = np.vstack([top, bottom])

        # labels
        for text, pos in [("Color", (10, 25)), ("Depth (aligned)", (W + 10, 25)),
                           ("Infrared", (10, H + 25))]:
            cv2.putText(grid, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("D405 Preview  [Q] quit", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
