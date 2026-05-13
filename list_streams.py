import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
    print()

    for sensor in dev.query_sensors():
        print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
        profiles = sensor.get_stream_profiles()
        seen = set()
        for p in profiles:
            vp = p.as_video_stream_profile()
            key = (p.stream_type(), p.format(), vp.width(), vp.height(), vp.fps())
            if key not in seen:
                seen.add(key)
                print(f"    {p.stream_type().name:10s} {p.format().name:10s} {vp.width()}x{vp.height()} @ {vp.fps()}fps")
        print()
