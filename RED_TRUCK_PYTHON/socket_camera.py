import socket
import time
import threading
import time
import cv2 
import base64
import threading
import pyrealsense2.pyrealsense2 as rs
import numpy as np

FRAME_WIDTH = 640 
FRAME_HEIGHT = 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)

print("[INFO] Starting streaming...")
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()    
align_to = rs.stream.color
align = rs.align(align_to)

 
# socket init
HOST = '192.168.133.1'
PORT = 65432
 
_socket = True
buffer = None

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen()
COLLISION = False
def camera():
    global buffer
    global _socket
    global COLLISION

    clipping_distance_in_meters = 1.5
    collision_percent = 0.15
    
    clipping_distance = clipping_distance_in_meters / depth_scale
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        frame = frames.get_color_frame()

        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        ir_frame_1 = frames.get_infrared_frame(1)
        ir_frame_2 = frames.get_infrared_frame(2)
        ir_frame_both = frames.get_infrared_frame()
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels

        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image_1, alpha=1), cv2.COLOR_BGRA2GRAY)

        ir_image_2 = np.asanyarray(ir_frame_2.get_data())
        ir_image_2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image_2, alpha=1), cv2.COLOR_BGRA2GRAY)

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        scaled_size = (frame.width, frame.height)
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(color_image, axis=0)

        frame = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_BGR2RGB)
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, ir_image_1)
        bg_removed[bg_removed != [0,0,0]] = 255 

        white_count = np.count_nonzero(bg_removed == 255)
        black_count = np.count_nonzero(bg_removed == 0)
        pixels = bg_removed.shape[0]*bg_removed.shape[1]*bg_removed.shape[2]
        if(white_count > pixels*collision_percent):
            COLLISION = True
        else:
            COLLISION = False
        cv2.imshow("bg_removed", bg_removed)
        t = cv2.waitKey(1)
        if t== ord('q'):
            s.close()
            break
        _, buffer = cv2.imencode('.jpg', bg_removed)
 
thread_camera = threading.Thread(target=camera).start()
#  socket init

while _socket:    
    conn, addr = s.accept()   
    try:
        if buffer is not None:
            print(addr)
            conn.sendall(base64.b64encode(buffer)+ bytes('#'+str(COLLISION), 'utf-8'))
    except Exception as e:
        print(e)
        conn.close() 

print("Terminating socket")
s.close()
exit(0)
