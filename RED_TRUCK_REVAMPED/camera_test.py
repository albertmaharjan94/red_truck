from utils import depth_camera
import cv2
import pyrealsense2.pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

FRAME_WIDTH=1280
FRAME_HEIGHT=720

print("[INFO] Enabling stream...")
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)

# profile
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# allign
align_to = rs.stream.color
align = rs.align(align_to)
print("[INFO] Camera ready.")

while True:
    frames= pipeline.wait_for_frames()
    t = cv2.waitKey(1)
    if t == ord('q'):
        break

    cv2.imshow("frame", frames.get_color_frame())

