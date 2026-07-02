import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
for i in range(10):
    frames = pipeline.wait_for_frames(timeout_ms=10000)
    print(f"Frame {i} OK")
pipeline.stop()
