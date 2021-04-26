import time
import serial
import pygame
import csv
import datetime
import sys

# print('This message will be displayed on the screen.')

original_stdout = sys.stdout

pygame.init()
pygame.display.set_mode()
pygame.key.set_repeat()
# ser = serial.Serial('COM6', 250000, timeout=1)
# ser.flush()	
SPEED = 0
import sys

DIRECTION= 230
# def writeArduiono(d, s):
#     ACTION = (str(s)+"#" +str(d)+ "\n").encode('utf_8')
#     # print(ACTION)
#     ser.write(ACTION)
#     line = ser.readline().decode('utf-8').rstrip()	
#     # print(line)
#  main
pygame.key.set_repeat()
write = False
count = 1

f = open('filename.txt', 'w')
sys.stdout = f # Change the standard output to the file we created.

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
            if event.key == ord('l'):
                if write == False:
                    write = True
                elif write == True:
                    write = False
            if event.key == ord('q'):
                f.close()
                sys.exit()

        if event.type == pygame.KEYUP:
            if event.key == ord('w') or event.key == ord('s'):
                SPEED = 0
            if event.key == ord('a') or event.key == ord('d'):
                DIRECTION = 230
        # print(DIRECTION, SPEED)
    # writeArduiono(DIRECTION, SPEED)
    if write == True:
        print(f"{SPEED}, {DIRECTION}")  