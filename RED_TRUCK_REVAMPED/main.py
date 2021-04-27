import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import math
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, K_q

import os
import serial
import threading as t
import time

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils
from utils import depth_camera
from utils import pygame_screen

import multiprocessing
import sys
import arduino_serial

# pygame init
pygame.init()

font = pygame.font.SysFont('timesnewroman',  22)

pygame.display.set_caption("OpenCV camera stream on Pygame")
# screen
X = pygame.display.Info().current_w
Y = pygame.display.Info().current_h

screen = pygame.display.set_mode([X, Y])

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


 # # download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
# print("[INFO] Loading model...")
# CONE_CKPT = "./frozen_inference_graph.pb"
SSD_CKPT = "./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"

ssd_graph = tf.compat.v1.Graph()
with ssd_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(SSD_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    session_rcnn = tf.compat.v1.Session(graph=ssd_graph)


# Input tensor is the image
image_tensor = ssd_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = ssd_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = ssd_graph.get_tensor_by_name('detection_scores:0')
detection_classes = ssd_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = ssd_graph.get_tensor_by_name('num_detections:0')
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


class Frame(object):
    def __init__(self):
        print("[INFO] Initializing frame...")

        self.status = dict({
            "collision": False, 
            "override": True,
            "playback": False,
            "speed": 0,
            "direction": 30
        })

        self.COLLISION = False
        self.PLAYBACK = False

        self.clipping_distance_in_meters = 2.5
        self.collision_percent = 0.15
        
        self.pipeline, self.config, self.depth_scale, self.align = depth_camera.create_pipeline() 
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # #  RC params
        self.SPEED = 0
        self.DIRECTION= 30
        # MID_ANGLE = 230
        # MAX_SPEED = 100
        # MIN_SPEED = 5

    # coordinate distance
    def distance(self, x1, x2, y1, y2):
        return math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )

    def draw_from_points(self,cv_image, points):
        """Takes the cv_image and points and draws a rectangle based on the points.
        Returns a cv_image."""
        for (x, y, w, h), n in points:
            cv.Rectangle(cv_image, (x, y), (x + w, y + h), 255)
        return cv_image
        
    

    def cvimage_to_pygame(self, image):
        """Convert cvimage into a pygame image"""
        image_rgb = cv2.CreateMat(image.height, image.width, cv2.CV_8UC3)
        cv2.CvtColor(image, image_rgb, cv2.CV_BGR2RGB)
        return pygame.image.frombuffer(image.tostring(), cv2.GetSize(image_rgb),
                                    "RGB")

    def run(self):
        print(self.pipeline, self.align, self.clipping_distance, self.collision_percent)
        with tf.compat.v1.Session(graph=ssd_graph) as session_rcnn:
            while True:
                # reference 
                # _frame = dict({
                #     "depth_image": None,
                #     "color_image": None,
                #     "ir_image_1": None,
                #     "ir_image_2": None,
                #     "bg_removed": None,
                #     "collision": False
                # })

                _frames = depth_camera.create_frame(self.pipeline,self.align, self.clipping_distance, self.collision_percent)
                self.status["collision"] = _frames["collision"]
                print(_frames)
                # events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); #sys.exit() if sys is imported
                    if event.type == pygame.KEYDOWN:
                        if event.key == ord('o'):
                            if self.status["override"] == True:
                                self.status["override"] = False
                            else:
                                self.status["override"] = True
                        if event.key == ord('w'):
                            self.status["speed"] = _speed.value = 6
                        elif event.key == ord('s'):
                            self.status["speed"] = _speed.value = -6
                        if event.key == ord('a'):
                            self.status["direction"] = _direction.value= 60
                        elif event.key == ord('d'):
                            self.status["direction"] =_direction.value= 0
                        if event.key == ord('l'):
                            if WRITE == False:
                                WRITE = True
                            elif WRITE == True:
                                WRITE = False

                        if event.key == ord('p'):
                            if self.PLAYBACK == None:
                                self.PLAYBACK = "Playing"
                            elif PLAYBACK != None:
                                self.PLAYBACK = None
                        if event.key == ord('q'):
                            sys.exit()

                    if event.type == pygame.KEYUP:
                        if event.key == ord('w') or event.key == ord('s'):
                            self.status["speed"] = 0
                        if event.key == ord('a') or event.key == ord('d'):
                            self.status["direction"] = 30

                # frame = _frames["color_image"].swapaxes(0, 1) 
                screen.fill([0, 0, 0])
                
                # init screen color image
                # screen.blit(pygame.surfarray.make_surface(cv2.cvtColor(_frames["color_image"], cv2.COLOR_BGR2RGB).swapaxes(0,1)), (0, 0))

                # with session_rcnn.as_default():
                #     # Perform the actual detection by running the model with the image as input
                #     (boxes, scores, classes, num) = session_rcnn.run([detection_boxes, detection_scores, detection_classes, num_detections],
                #                                                 feed_dict={image_tensor: np.expand_dims(_frames["color_image"], axis=0)})
                    
                #     boxes = np.squeeze(boxes)
                #     classes = np.squeeze(classes).astype(np.int32)
                #     scores = np.squeeze(scores)
                #     for idx in range(int(num)):
                #         class_ = classes[idx]
                #         score = scores[idx]
                #         box = boxes[idx]

                #         # print(class_,classes_90[class_])
                #         if class_ not in colors_hash:
                #             colors_hash[class_] = tuple(np.random.choice(range(256), size=3))

                #         if score > 0.3 and class_ in [1,2,3,4,5,6,7,8,10,11,13,14,15,16,17]:
                #             # print(class_)
                #             left = int(box[1] * FRAME_WIDTH)
                #             top = int(box[0] * FRAME_HEIGHT)
                #             right = int(box[3] * FRAME_WIDTH)
                #             bottom = int(box[2] * FRAME_HEIGHT)

                #             p1 = (left, top)
                #             p2 = (right, bottom)
                #             pygame.draw.rect(screen, (0,0,0), (p1, p2), 5)

                #             screen.blit(font.render(classes_90[class_], True, (100,255,0), (0,0,0)), 
                #                 p1
                #             )

                # normal 
                # _pygame_overlay = pygame_screen.create_pyframe(
                #     screen, _frames, FRAME_HEIGHT, FRAME_WIDTH, self.status, font
                # )
                # _pygame_overlay.blit(screen, (0,0))
                
                # update this
                pygame.display.update() 
        
        _running.value = False

        self.pipeline.stop()
        cv2.destroyAllWindows()
def main():
    frame = Frame()
    _arduino = arduino_serial.Arduino()

    RUNNING = multiprocessing.Value("b", True)
    SPEED = multiprocessing.Value("d", 0)
    DIRECTION = multiprocessing.Value("d", 30)

    p_camera = multiprocessing.Process(target=frame.run, args=(RUNNING, SPEED, DIRECTION,))
    # p_arduino = multiprocessing.Process(target=_arduino.arduino_process, args=(RUNNING, SPEED, DIRECTION,))
    
    
    p_camera.start()
    # p_arduino.start()


    p_camera.join()
    # p_arduino.join()


if __name__ == '__main__':
    main()






# # detection_graph = tf_utils.load_model(CONE_CKPT)


# # get the middle of 2 boudries
# def mid_from_boundries_clock(left_cone, right_cone):
#     if left_cone is None:
#         left_cone = ((0,FRAME_HEIGHT/2), None)
        
#     if right_cone is None:
#         right_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None)

#     if left_cone is not None and left_cone[1] is not None and left_cone[1] =="ORANGE":
#         if right_cone[1] is None or (right_cone[1] is not None and right_cone[1] =="ORANGE"):
#             right_cone = ((0, FRAME_HEIGHT/2), None)
#         else:
#             left_cone = ((0,FRAME_HEIGHT/2), None)
    
#     if right_cone[1] is not None and right_cone[1] =="GREEN":
#         if left_cone[1] is None or (left_cone[1] is not None and left_cone[1] =="GREEN"):
#             left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None)
#         else:
#             right_cone = ((FRAME_WIDTH,FRAME_HEIGHT/2), None)
#     #  middle of two objects
#     _mid = (left_cone[0][0]+right_cone[0][0])/2
#     return _mid

# def mid_from_boundries_anti_clock(left_cone, right_cone):
#     if left_cone is None:
#         left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
        
#     if right_cone is None:
#         right_cone = ((0,FRAME_HEIGHT/2), None)

#     if left_cone[1] is not None and left_cone[1] =="ORANGE":
#         if right_cone[1] is None or (right_cone[1] is not None and right_cone[1] =="ORANGE"):
#             right_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
#         else:
#             left_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None) 
    
#     if right_cone[1] is not None and right_cone[1] =="GREEN":
#         if left_cone[1] is None or (left_cone[1] is not None and left_cone[1] =="GREEN"):
#             left_cone = ((0,FRAME_HEIGHT/2), None)
#         else:
#             right_cone = ((0,FRAME_HEIGHT/2), None)
#     #  middle of two objects
#     _mid = (left_cone[0][0]+right_cone[0][0])/2
#     return _mid



# with tf.compat.v1.Session(graph=detection_graph) as sess:
#     while True:
#         _, _front_camera = cap.read()
#         # if CAR_STATE[1] == "STRAIGHT":
#         #     frames = pipeline.wait_for_frames()
#         # else:
#         #     frames = pipeline_back.wait_for_frames()
        
        
#         _tmp_speed = 0
#         _tmp_direction = 230

        
#         with detection_graph.as_default():
            
#             if CAR_STATE[1] == "STRAIGHT":
#                 crops, crops_coordinates = ops.extract_crops(
#                         _front_camera, FRAME_HEIGHT, FRAME_WIDTH,
#                         FRAME_HEIGHT-20, FRAME_HEIGHT-20)
#             else:
#                 crops, crops_coordinates = ops.extract_crops(
#                         frame, FRAME_HEIGHT, FRAME_WIDTH,
#                         FRAME_HEIGHT-20, FRAME_HEIGHT-20)
            
#             detection_dict = tf_utils.run_inference_for_batch(crops, sess)

#             # The detection boxes obtained are relative to each crop. Get
#             # boxes relative to the original image
#             # IMPORTANT! The boxes coordinates are in the following order:
#             # (ymin, xmin, ymax, xmax)
#             boxes = []
#             for box_absolute, boxes_relative in zip(
#                     crops_coordinates, detection_dict['detection_boxes']):
#                 boxes.extend(ops.get_absolute_boxes(
#                     box_absolute,
#                     boxes_relative[np.any(boxes_relative, axis=1)]))
#             if boxes:
#                 boxes = np.vstack(boxes)

#             # Remove overlapping boxes
#             boxes = ops.non_max_suppression_fast(
#                 boxes, NON_MAX_SUPPRESSION_THRESHOLD)

#             # Get scores to display them on top of each detection
#             boxes_scores = detection_dict['detection_scores']
#             boxes_scores = boxes_scores[np.nonzero(boxes_scores)]
#             detected = False
#             hasLeft = False
#             hasRight = False
            

#             right_cone = None
#             left_cone = None
            
#             for box, score in zip(boxes, boxes_scores):
#                 if score > 0.3:
#                     left = int(box[1])
#                     top = int(box[0])
#                     right = int(box[3])
#                     bottom = int(box[2])

#                     # center of object
#                     avg_x = (left+right)/2
#                     avg_y = (top+bottom)/2

#                     # find the area of the object box
#                     width = distance(left, right, top, bottom)
#                     height = distance(left, right, top, bottom)                
#                     area = int((width * height)/100)


#                     # motor control only if area of the object is in between two values
#                     if CAR_STATE[1] == "STRAIGHT":
#                         min_a = 300
#                         max_a = 2500
#                     else:
#                         min_a = 50
#                         max_a = 2500

#                     if(area > min_a and area < max_a):
#                         p1 = (left, top)
#                         p2 = (right, bottom)


#                         r,g,b = cv_utils.predominant_rgb_color(
#                                 frame, top, left, bottom, right)
#                         _color = None
#                         if(g == 255):
#                             _color ="GREEN"
#                         else:
#                             _color = "ORANGE"

#                         # dominant_color = cv_utils.predominant_rgb_color(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV))

#                         if((avg_x  > LEFT_START_POINT[0] and avg_x < RIGHT_START_POINT[0]) 
#                             or (avg_y > LEFT_START_POINT[1] and avg_y < RIGHT_START_POINT[1]) ):
#                             detected = True

#                         if(avg_x  < (FRAME_WIDTH/2)):
#                             cone = "LEFT"
#                         else:
#                             cone = "RIGHT"

#                         if(cone == "LEFT" and _color == "GREEN"):
#                             if(hasLeft):
#                                 pass
#                             else:
#                                 hasLeft = True
#                                 left_cone = (((right+right)/2, (top+bottom)/2),_color)

#                                 cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
#                                 cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
#                                     1, (b,g,r), 2, cv2.LINE_AA) 

#                         if(cone == "RIGHT"  and _color == "ORANGE"):
#                             if(hasRight):
#                                 pass
#                             else:
#                                 hasRight = True
#                                 right_cone = (((right+right)/2, (top+bottom)/2), _color)

#                                 cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
#                                 cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
#                                     1, (b,g,r), 2, cv2.LINE_AA) 
                        
#                         if(cone=="LEFT" and hasLeft == False and _color == "ORANGE"):
#                             hasLeft = True
#                             left_cone = (((right+right)/2, (top+bottom)/2),_color)
#                             cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
#                             cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
#                                 1, (b,g,r), 2, cv2.LINE_AA) 

#                         if(cone=="RIGHT" and hasRight == False and _color == "GREEN"):
#                             hasRight = True
#                             right_cone = (((left+left)/2, (top+bottom)/2),_color)

#                             cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
#                             cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
#                                 1, (b,g,r), 2, cv2.LINE_AA) 
                    

#                 CENTER_X = (int(FRAME_WIDTH/2))
#                 if CAR_STATE[0] == "CLOCK":
#                 #  middle of two objects
#                     _mid = mid_from_boundries_clock(left_cone, right_cone)
#                 elif CAR_STATE[0] == "ANTI-CLOCK":
#                     _mid = mid_from_boundries_anti_clock(left_cone, right_cone)

                
#         if(detected):
#             LINE_COLOR = (0,0,255)
#         else:
#             LINE_COLOR = (0,255,0)
#         # print(LINE_COLOR)
#         cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 20, (255,255,0), 2)
#         cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
#         cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        
#         # debugging text
#         cv2.putText(frame,f"Overide State: {OVERRIDE}", (int(FRAME_WIDTH/1.3)-20,int(FRAME_HEIGHT/2)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"Detected: {detected}", (int(FRAME_WIDTH/1.3)-20, int(FRAME_HEIGHT/2)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"Collsion: {collision}", (int(FRAME_WIDTH/1.3)-20, int(FRAME_HEIGHT/2)+10),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        

#         cv2.putText(frame,f"Left Cone: {left_cone}", (10,int(FRAME_HEIGHT/2)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"Right Cone: {right_cone}", (10,int(FRAME_HEIGHT/2)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"Face: {CAR_STATE[0]}", (10,int(FRAME_HEIGHT)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"DRIVE: {CAR_STATE[1]}", (10,int(FRAME_HEIGHT)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


#         cv2.putText(frame,f"SPEED: {SPEED}", (FRAME_WIDTH-250, FRAME_HEIGHT-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         cv2.putText(frame,f"DIRECTION: {DIRECTION}", (FRAME_WIDTH-250,FRAME_HEIGHT-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
#         try:
#             cv2.putText(frame,f"MID: {_mid}", (FRAME_WIDTH-250,FRAME_HEIGHT-100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
#             cv2.line(frame, (int(_mid), 0), (int(_mid), FRAME_HEIGHT), (100,200,200), LINE_THICKNESS) 
#         except:
#             continue
        
#         t = cv2.waitKey(1) & 0xFF
#         if t == ord('q'):
#             break
        
#         if t == ord('i'):
#             CAR_STATE[1]="STRAIGHT"
#         elif t == ord('k'):
#             CAR_STATE[1] = "REVERSE"

#         if t == ord('j'):
#             CAR_STATE[0]="CLOCK"
#         elif t == ord('l'):
#             CAR_STATE[0] = "ANTI-CLOCK"


#         if t == ord('o'):
#             if OVERRIDE == False:
#                 OVERRIDE = True
#                 _tmp_speed = 0
#                 print("Overide ON")
#             else:
#                 if collision == True:
#                     _tmp_speed = 0
#                 else:
#                     _tmp_speed = 50
#                 OVERRIDE = False
#                 print("Overdie OFF")
       

#         if(OVERRIDE == False):
#             if(collision == True):
#                 _tmp_speed = 0
#             else:
#                 ## alpha stage of motor and speed control
#                 # if((_mid < CENTER_X and _mid > LEFT_START_POINT[0])):   
#                 #     DIRECTION = np.interp(_mid,[320,510],[30,60])
#                 #     # DIRECTION = 60
#                 # elif((_mid > CENTER_X and _mid < RIGHT_START_POINT[0]) or left_cone[1] == "ORANGE"):
#                 #     DIRECTION = np.interp(_mid,[125,320],[0,30])
#                 # else:
#                 #     DIRECTION = 230
#                 # SPEED = 10

#                 # dynamic direction and speed of motor
#                 _tmp_direction =  int(np.interp(_mid,[LEFT_START_POINT[0],RIGHT_START_POINT[0]],[260,200]))
#                 # middle angle is 30
#                 diff_angle = abs(_tmp_direction-MID_ANGLE)

#                 if (CAR_STATE[1] == "STRAIGHT"):
#                     # SPEED  = int(np.interp(diff_angle, [0, MID_ANGLE],[MAX_SPEED, MIN_SPEED]))
#                     _tmp_speed = 50
#                 elif(CAR_STATE[1] =="REVERSE"):
#                     # SPEED  = -int(np.interp(diff_angle, [0, MID_ANGLE],[MAX_SPEED, MIN_SPEED]))
#                     _tmp_speed = -50
#         else:
#             _tmp_speed = 0
#             _tmp_direction =  230


         
#         if OVERRIDE == True:    
#             if t == ord('w'):
#                 _tmp_speed = 50
#             elif t == ord('s'):
#                 _tmp_speed = 0
#             elif t == ord('x'):
#                 _tmp_speed = -50
#             else:
#                 _tmp_speed = 0
                  
#             if t == ord('a'):
#                 _tmp_direction = 260
#             elif t == ord('d'):
#                 _tmp_direction = 200



#         SPEED = _tmp_speed
#         DIRECTION = _tmp_direction


        
#         frame_final = frame.copy()
#         # print(collision)
#         # overlay shit here 
#         overlay_aspect = [int(FRAME_HEIGHT*0.2), int(FRAME_WIDTH*0.2)]
#         w_overlay_start = 100
#         frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(depth_colormap, (overlay_aspect[1], overlay_aspect[0]))
#         w_overlay_start += overlay_aspect[1] + 20
#         frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(ir_image_1, (overlay_aspect[1], overlay_aspect[0]))
#         w_overlay_start += overlay_aspect[1] + 20
#         frame_final[10:10+overlay_aspect[0],w_overlay_start:w_overlay_start+overlay_aspect[1],:] = cv2.resize(bg_removed, (overlay_aspect[1], overlay_aspect[0]))

#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

#         cv2.imshow('RealSense', frame_final)

#         hasStarted = True



# hasStarted = False

# print("[INFO] stop streaming ...")
# cap.release()
# cv2.destroyAllWindows()
# pipeline.stop()
# pipeline_back.stop()
# print("[INFO] closing thread ...")
# motorThread.join()

# # pipeline.stop()
# raise SystemExit

