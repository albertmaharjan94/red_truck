import pygame
import cv2
import numpy as np


def create_pyframe(screen, _frames, FRAME_HEIGHT, FRAME_WIDTH, status, font):
    
    overlay_aspect = [int(FRAME_HEIGHT*0.2), int(FRAME_WIDTH*0.2)]
    w_overlay_start = 100
    
    # screen.blit(pygame.surfarray.make_surface(cv2.cvtColor(_frames["color_image"], cv2.COLOR_BGR2RGB).swapaxes(0,1)), (0, 0))

    # depth colormap
    screen.blit(pygame.surfarray.make_surface(
        cv2.resize(
            cv2.cvtColor(_frames["depth_image"], cv2.COLOR_BGR2RGB).swapaxes(0,1), 
            (overlay_aspect[0], overlay_aspect[1]))
    ), (w_overlay_start,30))
    screen.blit(font.render('Depth', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
    w_overlay_start += overlay_aspect[1] + 20

    # Ir image 1
    screen.blit(pygame.surfarray.make_surface(
        cv2.resize(
            cv2.cvtColor(_frames["ir_image_1"], cv2.COLOR_BGR2RGB).swapaxes(0,1), 
            (overlay_aspect[0], overlay_aspect[1]))
    ), (w_overlay_start,30))
    screen.blit(font.render('IR 1', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
    w_overlay_start += overlay_aspect[1] + 20

    # Ir image 2
    screen.blit(pygame.surfarray.make_surface(
        cv2.resize(
            cv2.cvtColor(_frames["ir_image_2"], cv2.COLOR_BGR2RGB).swapaxes(0,1), 
            (overlay_aspect[0], overlay_aspect[1]))
    ), (w_overlay_start,30))
    screen.blit(font.render('IR 2', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))
    w_overlay_start += overlay_aspect[1] + 20

    # BG removed
    screen.blit(pygame.surfarray.make_surface(
        cv2.resize(_frames["bg_removed"].swapaxes(0,1), (overlay_aspect[0], overlay_aspect[1]))
    ), (w_overlay_start,30))
    screen.blit(font.render('BG Rem', True, (100,255,0), (0,0,0)), (w_overlay_start,overlay_aspect[0]))

    # dummy text
    status_font = pygame.font.SysFont('timesnewroman',  25)
    status_y = FRAME_HEIGHT
    screen.blit(
        status_font.render('Status', True, (255,255,255))
    , (0,status_y))
    status_y = status_y + 25
    screen.blit(
        status_font.render(f'Override: {status["override"]}', True, (255,255,255))
    , (0,status_y))
    status_y = status_y + 25
    screen.blit(
        status_font.render(f'Collision: {status["collision"]}', True, (255,255,255))
    , (0,status_y))
    status_y = status_y + 25
    screen.blit(
        status_font.render(f'Playback: {status["playback"]}', True, (255,255,255))
    , (0,status_y))

    status_y = status_y + 25
    screen.blit(
        status_font.render(f'Speed: {status["speed"]}', True, (255,255,255))
    , (0,status_y))

    status_y = status_y + 25
    screen.blit(
        status_font.render(f'Direction: {status["direction"]}', True, (255,255,255))
    , (0,status_y))

    return screen
    