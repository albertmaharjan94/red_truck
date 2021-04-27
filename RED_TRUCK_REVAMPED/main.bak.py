import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import math

import os
import serial
import threading as t
import time

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

import socket
import base64
import threading
 

ser = serial.Serial('/dev/ttyUSB0', 250000, timeout=1.5)
time.sleep(3)
ser.flush()



CAR_STATE = ["CLOCK","STRAIGHT"]


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

OFFSET = 450


DETECT_MIN = (int(FRAME_WIDTH/2)-OFFSET, 0)
DETECT_MAX = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)


LEFT_START_POINT = (int(FRAME_WIDTH/2)-OFFSET, 0) 
LEFT_END_POINT = (int(FRAME_WIDTH/2)-OFFSET, FRAME_HEIGHT)

RIGHT_START_POINT = (int(FRAME_WIDTH/2)+OFFSET, 0) 
RIGHT_END_POINT = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)

LINE_COLOR = (0, 0, 255) 

LINE_THICKNESS = 5

SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5

# collision params
clipping_distance_in_meters = 2.5
collision_percent = 0.15
collision = False

OVERRIDE = True
# Configure depth front          
pipeline = rs.pipeline()
config = rs.config()
# 035422073295
config.enable_device('035422073295')
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)

# configure depth back
# 034422074343
pipeline_back = rs.pipeline()
config_back = rs.config()
config_back.enable_device('034422074343')
config_back.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config_back.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config_back.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
config_back.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)


print("[INFO] Starting streaming...")
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()


profile_back = pipeline_back.start(config_back)
depth_sensor_back = profile_back.get_device().first_depth_sensor()
depth_scale_back = depth_sensor_back.get_depth_scale()

clipping_distance = clipping_distance_in_meters / depth_scale
clipping_distance_back = clipping_distance_in_meters / depth_scale_back

print("Depth Scale is: " , depth_scale)
print("Depth Scale is: " , depth_scale_back)
print("Depth Scale is: " , clipping_distance)
print("Depth Scale is: " , clipping_distance_back)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


print("[INFO] Camera ready.")

# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
print("[INFO] Loading model...")
CONE_CKPT = "./frozen_inference_graph.pb"
RCNN_CKPT = "./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(CONE_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

rcnn_graph = tf.compat.v1.Graph()
with rcnn_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(RCNN_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    session_rcnn = tf.compat.v1.Session(graph=rcnn_graph)

# coordinate distance
def distance(x1, x2, y1, y2):
    # width =math.sqrt( ((xmin-ymin)**2)+((xmax-ymin)**2) )
    # height = math.sqrt( ((xmax-ymin)**2)+((xmax-ymax)**2) )
    # area = int((width * height)/100)
    return math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )


#  RC params
SPEED = 0
DIRECTION= 230
MID_ANGLE = 230
MAX_SPEED = 100
MIN_SPEED = 5

detection_graph = tf_utils.load_model(CONE_CKPT)
rcnn_graph = tf_utils.load_model(RCNN_CKPT)

#  camera
cap = cv2.VideoCapture(6)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

hasStarted = False

# input to arduino 
def writeArduiono():
    time.sleep(5)
    while True:
        global SPEED
        global DIRECTION
        if hasStarted:
            # print(f"DIRECTION: {DIRECTION}, SPEED: {SPEED}")
            if DIRECTION == 200 or DIRECTION == 290:
                ACTION = (str(SPEED)+"#" +str(DIRECTION)+ "\n").encode('utf_8')
                ser.write(ACTION)
                try:
                    line = ser.readline().decode('utf-8').rstrip()	
                    print(line)
                except Exception as e:
                    print(e)
            else:
                ACTION = (str(SPEED)+"#" +str(DIRECTION)+ "\n").encode('utf_8')
                ser.write(ACTION)
                try:
                    line = ser.readline().decode('utf-8').rstrip()	
                    print(line)
                except Exception as e:
                    print(e)

# start motor thread for individual process
motorThread = t.Thread(target = writeArduiono)
motorThread.start()



# get the middle of 2 boudries
def mid_from_boundries_clock(left_cone, right_cone):
    if left_cone is None:
        left_cone = ((0,FRAME_HEIGHT/2), None)
        
    if right_cone is None:
        right_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None)

    if left_cone is not None and left_cone[1] is not None and left_cone[1] =="ORANGE":
        if right_cone[1] is None or (right_cone[1] is not None and right_cone[1] =="ORANGE"):
            right_cone = ((0, FRAME_HEIGHT/2), None)
        else:
            left_cone = ((0,FRAME_HEIGHT/2), None)
    
    if right_cone[1] is not None and right_cone[1] =="GREEN":
        if left_cone[1] is None or (left_cone[1] is not None and left_cone[1] =="GREEN"):
            left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None)
        else:
            right_cone = ((FRAME_WIDTH,FRAME_HEIGHT/2), None)
    #  middle of two objects
    _mid = (left_cone[0][0]+right_cone[0][0])/2
    return _mid

def mid_from_boundries_anti_clock(left_cone, right_cone):
    if left_cone is None:
        left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
        
    if right_cone is None:
        right_cone = ((0,FRAME_HEIGHT/2), None)

    if left_cone[1] is not None and left_cone[1] =="ORANGE":
        if right_cone[1] is None or (right_cone[1] is not None and right_cone[1] =="ORANGE"):
            right_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
        else:
            left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
    
    if right_cone[1] is not None and right_cone[1] =="GREEN":
        if left_cone[1] is None or (left_cone[1] is not None and left_cone[1] =="GREEN"):
            left_cone = ((0,FRAME_HEIGHT/2), None)
        else:
            right_cone = ((0,FRAME_HEIGHT/2), None)
    #  middle of two objects
    _mid = (left_cone[0][0]+right_cone[0][0])/2
    return _mid

# Input tensor is the image
image_tensor = rcnn_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = rcnn_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = rcnn_graph.get_tensor_by_name('detection_scores:0')
detection_classes = rcnn_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = rcnn_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
print("[INFO] Model loaded.")
colors_hash = {}
# 1,2,3,4,5,6,7,8,10,11,13,14,15,16,17
classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "cone" ] 

def convert_image(i):
    m = np.min(i)
    M = np.max(i)
    i = np.divide(i, np.array([M - m], dtype=np.float)).astype(np.float)
    i = (i - m).astype(np.float)
    i8 = (i * 255.0).astype(np.uint8)
    if i8.ndim == 3:
        i8 = cv2.cvtColor(i8, cv2.COLOR_BGRA2GRAY)
    i8 = cv2.equalizeHist(i8)
    colorized = cv2.applyColorMap(i8, cv2.COLORMAP_JET)
    colorized[i8 == int(m)] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    m = float("{:.2f}".format(m))
    M = float("{:.2f}".format(M))
    return colorized


with tf.compat.v1.Session(graph=detection_graph) as sess,tf.compat.v1.Session(graph=rcnn_graph) as session_rcnn:
    while True:
        _, _front_camera = cap.read()
        if CAR_STATE[1] == "STRAIGHT":
            frames = pipeline.wait_for_frames()
        else:
            frames = pipeline_back.wait_for_frames()
        
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # color_frame = frames.get_color_frame()
        frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

    
        ir_frame_1 = frames.get_infrared_frame(1)
        # ir_frame_2 = frames.get_infrared_frame(2)
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels



        

        # if not ir_frame_1 or not ir_frame_2 or depth_frame or not frame:
        #     continue

        # Convert images to numpy arrays

        # ir_image_both = np.asanyarray(ir_frame_both.get_data())
        # ir_image_both = convert_image(ir_image_both)


        ir_image_1 = np.asanyarray(ir_frame_1.get_data())
        ir_image_1 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image_1, alpha=1), cv2.COLOR_BGRA2GRAY)

        # ir_image_2 = np.asanyarray(ir_frame_2.get_data())
        # ir_image_2 = cv2.applyColorMap(cv2.convertScaleAbs(ir_image_2, alpha=1), cv2.COLOR_BGRA2GRAY)


        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        scaled_size = (frame.width, frame.height)
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(color_image, axis=0)

        frame = np.asanyarray(frame.get_data())

        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, ir_image_1)

        bg_removed[bg_removed != [0,0,0]] = 255 

        white_count = np.count_nonzero(bg_removed == 255)
        black_count = np.count_nonzero(bg_removed == 0)
        pixels = bg_removed.shape[0]*bg_removed.shape[1]*bg_removed.shape[2]
        if(white_count > pixels*collision_percent):
            collision = True
        else:
            collision = False
        
        _tmp_speed = 0
        _tmp_direction = 230

        with session_rcnn.as_default():
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = session_rcnn.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                        feed_dict={image_tensor: image_expanded})
            
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            for idx in range(int(num)):
                class_ = classes[idx]
                score = scores[idx]
                box = boxes[idx]

                # print(class_,classes_90[class_])
                if class_ not in colors_hash:
                    colors_hash[class_] = tuple(np.random.choice(range(256), size=3))

                if score > 0.7 and class_ in [1,2,3,4,5,6,7,8,10,11,13,14,15,16,17]:
                    left = int(box[1] * FRAME_WIDTH)
                    top = int(box[0] * FRAME_HEIGHT)
                    right = int(box[3] * FRAME_WIDTH)
                    bottom = int(box[2] * FRAME_HEIGHT)

                    r,g,b = cv_utils.predominant_rgb_color_object(
                                depth_colormap, top, left, bottom, right)
                    

                    p1 = (left, top)
                    p2 = (right, bottom)
                    # r, g, b = colors_hash[class_]
                    cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                    cv2.putText(frame, classes_90[class_], p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255,0,0), 2, cv2.LINE_AA)


        with detection_graph.as_default():
            crops, crops_coordinates = ops.extract_crops(
                        _front_camera, FRAME_HEIGHT, FRAME_WIDTH,
                        FRAME_HEIGHT-20, FRAME_HEIGHT-20)
            
            detection_dict = tf_utils.run_inference_for_batch(crops, sess)

            # The detection boxes obtained are relative to each crop. Get
            # boxes relative to the original image
            # IMPORTANT! The boxes coordinates are in the following order:
            # (ymin, xmin, ymax, xmax)
            boxes = []
            for box_absolute, boxes_relative in zip(
                    crops_coordinates, detection_dict['detection_boxes']):
                boxes.extend(ops.get_absolute_boxes(
                    box_absolute,
                    boxes_relative[np.any(boxes_relative, axis=1)]))
            if boxes:
                boxes = np.vstack(boxes)

            # Remove overlapping boxes
            boxes = ops.non_max_suppression_fast(
                boxes, NON_MAX_SUPPRESSION_THRESHOLD)

            # Get scores to display them on top of each detection
            boxes_scores = detection_dict['detection_scores']
            boxes_scores = boxes_scores[np.nonzero(boxes_scores)]
            detected = False
            hasLeft = False
            hasRight = False
            

            right_cone = None
            left_cone = None
            
            for box, score in zip(boxes, boxes_scores):
                if score > 0.1:
                    left = int(box[1])
                    top = int(box[0])
                    right = int(box[3])
                    bottom = int(box[2])

                    # center of object
                    avg_x = (left+right)/2
                    avg_y = (top+bottom)/2

                    # find the area of the object box
                    width = distance(left, right, top, bottom)
                    height = distance(left, right, top, bottom)                
                    area = int((width * height)/100)


                    # motor control only if area of the object is in between two values
                    if CAR_STATE[1] == "STRAIGHT":
                        min_a = 10
                        max_a = 2000
                    else:
                        min_a = 0
                        max_a = 2000

                    if(area > min_a and area < max_a):
                        p1 = (left, top)
                        p2 = (right, bottom)


                        r,g,b = cv_utils.predominant_rgb_color(
                                frame, top, left, bottom, right)
                        _color = None
                        if(g == 255):
                            _color ="GREEN"
                        elif (b == 255):
                            _color = "BLUE"
                        else:
                            _color = "ORANGE"

                        # dominant_color = cv_utils.predominant_rgb_color(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV))

                        if((avg_x  > LEFT_START_POINT[0] and avg_x < RIGHT_START_POINT[0]) 
                            or (avg_y > LEFT_START_POINT[1] and avg_y < RIGHT_START_POINT[1]) ):
                            detected = True

                        if(avg_x  < (FRAME_WIDTH/2)):
                            cone = "LEFT"
                        else:
                            cone = "RIGHT"

                        if(cone == "LEFT" and _color == "GREEN"):
                            if(hasLeft):
                                pass
                            else:
                                hasLeft = True
                                left_cone = (((right+right)/2, (top+bottom)/2),_color)

                                cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                                cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                    1, (b,g,r), 2, cv2.LINE_AA) 

                        if(cone == "RIGHT"  and _color == "ORANGE"):
                            if(hasRight):
                                pass
                            else:
                                hasRight = True
                                right_cone = (((right+right)/2, (top+bottom)/2), _color)

                                cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                                cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                    1, (b,g,r), 2, cv2.LINE_AA) 
                        
                        if(cone=="LEFT" and hasLeft == False and _color == "ORANGE"):
                            hasLeft = True
                            left_cone = (((right+right)/2, (top+bottom)/2),_color)
                            cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                            cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                1, (b,g,r), 2, cv2.LINE_AA) 

                        if(cone=="RIGHT" and hasRight == False and _color == "GREEN"):
                            hasRight = True
                            right_cone = (((left+left)/2, (top+bottom)/2),_color)

                            cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                            cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                1, (b,g,r), 2, cv2.LINE_AA) 
                    

                CENTER_X = (int(FRAME_WIDTH/2))
                if CAR_STATE[0] == "CLOCK":
                #  middle of two objects
                    _mid = mid_from_boundries_clock(left_cone, right_cone)
                elif CAR_STATE[0] == "ANTI-CLOCK":
                    _mid = mid_from_boundries_anti_clock(left_cone, right_cone)

                
        if(detected):
            LINE_COLOR = (0,0,255)
        else:
            LINE_COLOR = (0,255,0)
        # print(LINE_COLOR)
        cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 20, (255,255,0), 2)
        cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        
        # debugging text
        cv2.putText(frame,f"Overide State: {OVERRIDE}", (int(FRAME_WIDTH/1.3)-20,int(FRAME_HEIGHT/2)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Detected: {detected}", (int(FRAME_WIDTH/1.3)-20, int(FRAME_HEIGHT/2)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Collsion: {collision}", (int(FRAME_WIDTH/1.3)-20, int(FRAME_HEIGHT/2)+10),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        

        cv2.putText(frame,f"Left Cone: {left_cone}", (10,int(FRAME_HEIGHT/2)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Right Cone: {right_cone}", (10,int(FRAME_HEIGHT/2)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Face: {CAR_STATE[0]}", (10,int(FRAME_HEIGHT)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"DRIVE: {CAR_STATE[1]}", (10,int(FRAME_HEIGHT)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


        cv2.putText(frame,f"SPEED: {SPEED}", (FRAME_WIDTH-250, FRAME_HEIGHT-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"DIRECTION: {DIRECTION}", (FRAME_WIDTH-250,FRAME_HEIGHT-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        try:
            cv2.putText(frame,f"MID: {_mid}", (FRAME_WIDTH-250,FRAME_HEIGHT-100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
            cv2.line(frame, (int(_mid), 0), (int(_mid), FRAME_HEIGHT), (100,200,200), LINE_THICKNESS) 
        except:
            continue
        
        t = cv2.waitKey(1) & 0xFF
        if t == ord('q'):
            break
        
        if t == ord('i'):
            CAR_STATE[1]="STRAIGHT"
        elif t == ord('k'):
            CAR_STATE[1] = "REVERSE"

        if t == ord('j'):
            CAR_STATE[0]="CLOCK"
        elif t == ord('l'):
            CAR_STATE[0] = "ANTI-CLOCK"


        if t == ord('o'):
            if OVERRIDE == False:
                OVERRIDE = True
                _tmp_speed = 0
                print("Overide ON")
            else:
                if collision == True:
                    _tmp_speed = 0
                else:
                    _tmp_speed = 50
                OVERRIDE = False
                print("Overdie OFF")
       

        if(OVERRIDE == False):
            if(collision == True):
                _tmp_speed = 0
            else:
                ## alpha stage of motor and speed control
                # if((_mid < CENTER_X and _mid > LEFT_START_POINT[0])):   
                #     DIRECTION = np.interp(_mid,[320,510],[30,60])
                #     # DIRECTION = 60
                # elif((_mid > CENTER_X and _mid < RIGHT_START_POINT[0]) or left_cone[1] == "ORANGE"):
                #     DIRECTION = np.interp(_mid,[125,320],[0,30])
                # else:
                #     DIRECTION = 230
                # SPEED = 10

                # dynamic direction and speed of motor
                _tmp_direction =  int(np.interp(_mid,[LEFT_START_POINT[0],RIGHT_START_POINT[0]],[260,200]))
                # middle angle is 30
                diff_angle = abs(DIRECTION-MID_ANGLE)

                if (CAR_STATE[1] == "STRAIGHT"):
                    # SPEED  = int(np.interp(diff_angle, [0, MID_ANGLE],[MAX_SPEED, MIN_SPEED]))
                    _tmp_speed = 50
                elif(CAR_STATE[1] =="REVERSE"):
                    # SPEED  = -int(np.interp(diff_angle, [0, MID_ANGLE],[MAX_SPEED, MIN_SPEED]))
                    _tmp_speed = -50
        else:
            _tmp_speed = 0
            _tmp_direction =  230


         
        # if OVERRIDE == True:    
        if t == ord('w'):
            _tmp_speed = 50
        elif t == ord('s'):
            _tmp_speed = 0
        elif t == ord('x'):
            _tmp_speed = -50
                
        if t == ord('a'):
            _tmp_direction = 260
        elif t == ord('d'):
            _tmp_direction = 200



        SPEED = _tmp_speed
        DIRECTION = _tmp_direction


        
        frame_final = frame.copy()
        # print(collision)
        # overlay shit here 
        overlay_aspect = [int(FRAME_HEIGHT*0.2), int(FRAME_WIDTH*0.2)]
        w_overlay_start = 100
        frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(depth_colormap, (overlay_aspect[1], overlay_aspect[0]))
        w_overlay_start += overlay_aspect[1] + 20
        frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(ir_image_1, (overlay_aspect[1], overlay_aspect[0]))
        w_overlay_start += overlay_aspect[1] + 20
        # frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(ir_image_2, (overlay_aspect[1], overlay_aspect[0]))
        # w_overlay_start += overlay_aspect[1] + 20
        frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(bg_removed, (overlay_aspect[1], overlay_aspect[0]))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('RealSense', frame_final)

        hasStarted = True



hasStarted = False

print("[INFO] stop streaming ...")
cap.release()
cv2.destroyAllWindows()
pipeline.stop()
pipeline_back.stop()
print("[INFO] closing thread ...")
motorThread.join()

# pipeline.stop()
raise SystemExit

