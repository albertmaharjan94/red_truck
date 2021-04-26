from pygame.locals import KEYDOWN, K_ESCAPE, K_q
import pygame
import cv2
import sys
import numpy as np
import pyrealsense2.pyrealsense2 as rs
from helpers import depth_utils
import time
import threading

pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
# screen = pygame.display.set_mode([640, 480])

# screen
X = pygame.display.Info().current_w
Y = pygame.display.Info().current_h

# camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

OFFSET = 450

screen = pygame.display.set_mode([X, Y])
font = pygame.font.SysFont('timesnewroman',  22)

SPEED = 0
DIRECTION= 30
WRITE = False


# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# collision params
clipping_distance_in_meters = 1.5
collision_percent = 0.15
COLLISION = False

# camera init
# Depth camera
DEPTH_ID1 = "035422073295"
pipeline, config, profile, align, depth_scale = depth_utils.getDepthPipeline(FRAME_WIDTH, FRAME_HEIGHT,DEPTH_ID1)

clipping_distance = clipping_distance_in_meters / depth_scale
 
OVERRIDE = True
PLAYBACK = None
RECORDING = False
TIME_BIAS = 1

FILE_NAME = str(int(time.time())) + ".load"
f = open(FILE_NAME, 'a+')

# def playback():
#     global PLAYBACK
#     lines = f.read().splitlines()
#     while True:
#         if(PLAYBACK != None):
#             lines = f.read().splitlines()
#             # while running:
#             for i in lines[::TIME_BIAS]:
#                 _split = i.split(', ')
#                 SPEED = _split[0]
#                 DIRECTION = _split[1]
#                 # ACTION = (str(SPEED)+"#" +str(DIRECTION)+ "\n").encode('utf_8')
#                 # ser.write(ACTION)
#                 # line = ser.readline().decode('utf-8').rstrip()	
#                 print(i)
#             PLAYBACK = None

# playback_thread = threading.Thread(target=playback).start()

# main thread
try:
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


        #  events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); #sys.exit() if sys is imported
            if event.type == pygame.KEYDOWN:
                if event.key == ord('o'):
                    if OVERRIDE == True:
                        OVERRIDE = False
                    else:
                        OVERRIDE = True
                if event.key == ord('w'):
                    SPEED = 6
                elif event.key == ord('s'):
                    SPEED = -6
                if event.key == ord('a'):
                    DIRECTION = 60
                elif event.key == ord('d'):
                    DIRECTION = 0
                if event.key == ord('l'):
                    if WRITE == False:
                        WRITE = True
                    elif WRITE == True:
                        WRITE = False

                if event.key == ord('p'):
                    if PLAYBACK == None:
                        PLAYBACK = "Playing"
                    elif PLAYBACK != None:
                        PLAYBACK = None
                if event.key == ord('q'):
                    sys.exit()

            if event.type == pygame.KEYUP:
                if event.key == ord('w') or event.key == ord('s'):
                    SPEED = 0
                if event.key == ord('a') or event.key == ord('d'):
                    DIRECTION = 30

        # write to frame
        frame = frame.swapaxes(0, 1) 
        overlay_aspect = [int(FRAME_HEIGHT*0.2), int(FRAME_WIDTH*0.2)]
        w_overlay_start = 100

        screen.fill([0, 0, 0])
        # normal 
        screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
        # depth colormap
        screen.blit(pygame.surfarray.make_surface(
            cv2.resize(
                cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB).swapaxes(0,1), 
                (overlay_aspect[0], overlay_aspect[1]))
        ), (w_overlay_start,30))
        screen.blit(font.render('Depth', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
        w_overlay_start += overlay_aspect[1] + 20
        # Ir image 1
        screen.blit(pygame.surfarray.make_surface(
            cv2.resize(
                cv2.cvtColor(ir_image_1, cv2.COLOR_BGR2RGB).swapaxes(0,1), 
                (overlay_aspect[0], overlay_aspect[1]))
        ), (w_overlay_start,30))
        screen.blit(font.render('IR 1', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
        w_overlay_start += overlay_aspect[1] + 20
        # Ir image 2
        screen.blit(pygame.surfarray.make_surface(
            cv2.resize(
                cv2.cvtColor(ir_image_2, cv2.COLOR_BGR2RGB).swapaxes(0,1), 
                (overlay_aspect[0], overlay_aspect[1]))
        ), (w_overlay_start,30))
        screen.blit(font.render('IR 2', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
        w_overlay_start += overlay_aspect[1] + 20
        # BG removed
        screen.blit(pygame.surfarray.make_surface(
            cv2.resize(bg_removed.swapaxes(0,1), (overlay_aspect[0], overlay_aspect[1]))
        ), (w_overlay_start,30))
        screen.blit
        screen.blit(font.render('BG Rem', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
        # dummy text
        status_font = pygame.font.SysFont('timesnewroman',  25)
        status_y = FRAME_HEIGHT
        screen.blit(
            status_font.render('Status', True, (255,255,255))
        , (0,status_y))
        status_y = status_y + 25
        screen.blit(
            status_font.render(f'Override: {OVERRIDE}', True, (255,255,255))
        , (0,status_y))
        status_y = status_y + 25
        screen.blit(
            status_font.render(f'Collision: {COLLISION}', True, (255,255,255))
        , (0,status_y))
        status_y = status_y + 25
        screen.blit(
            status_font.render(f'Playback: {PLAYBACK}', True, (255,255,255))
        , (0,status_y))
        
        if WRITE == True:
            sys.stdout = f
            print(f"{SPEED}, {DIRECTION}")  
        else:
            sys.stdout = sys.__stdout__
        pygame.display.update()

except (KeyboardInterrupt, SystemExit):
    pygame.quit()
    pipeline.stop()
    cv2.destroyAllWindows()
    