import cv2
import multiprocessing
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import time
import ctypes
from ctypes import c_char_p
import base64
import socket

cap = cv2.VideoCapture(1)

def camera_process(_running, _collision, _buffer):
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
            _collision.value = True
        else:
            _collision.value = False

        cv2.imshow("bg_removed", bg_removed)
        t = cv2.waitKey(1)
        if t== ord('q'):
            _running.value = False
            break
        # _, _buffer.value = cv2.imencode('.jpg', bg_removed)
        print(type(base64.b64encode(cv2.imencode('.jpg', bg_removed)[1])))
        try:
            _buffer.value = str(base64.b64encode(cv2.imencode('.jpg', bg_removed)[1]))
        except Exception as e:
            print(e)


def dummp_process(_running, _collision, _buffer):
    # socket init
    HOST = '172.25.0.48'
    PORT = 65432
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    while _running.value:
        print(_buffer.value )
       
        continue

if __name__ == '__main__':
    # shared variable
    COLLISION = multiprocessing.Value("b", False)
    RUNNING = multiprocessing.Value("b", True)
    BUFFER = multiprocessing.Value(ctypes.c_wchar_p, 'TEST')

    p_camera = multiprocessing.Process(target=camera_process, args=(RUNNING, COLLISION,BUFFER,))
    p_dump = multiprocessing.Process(target=dummp_process, args=(RUNNING,COLLISION,BUFFER,))
    p_camera.start()
    p_dump.start()

    p_camera.join()
    p_dump.join()
    print(COLLISION.value)
        
    cap.release()
    cv2.destroyAllWindows()

