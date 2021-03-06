import time
import serial
import pygame
import sys
pygame.init()
pygame.display.set_mode()
pygame.key.set_repeat()

ser = serial.Serial('COM8', 250000, timeout=1.5)
ser.flush()	
SPEED = 0

DIRECTION= 230
def writeArduiono(d, s):
    ACTION = (str(s)+"," +str(d)+ "#").encode('ascii')
    ser.write(ACTION)
    line = ser.readline().decode('ascii').rstrip()	
    ser.flush()
    print(line)
#  main
pygame.key.set_repeat()
while True: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:
            if event.key == ord('w'):
                SPEED = 50
            elif event.key == ord('s'):
                SPEED = -50
            if event.key == ord('a'):
                DIRECTION = 260
            elif event.key == ord('d'):
                DIRECTION = 200
            elif event.key == ord('q'):
                sys.exit()
        if event.type == pygame.KEYUP:
            if event.key == ord('w') or event.key == ord('s'):
                SPEED = 0
            if event.key == ord('a') or event.key == ord('d'):
                DIRECTION = 230
    writeArduiono(DIRECTION, SPEED)