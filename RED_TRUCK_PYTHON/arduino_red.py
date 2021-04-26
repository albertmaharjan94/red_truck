import time
import serial
import pygame
import csv
import datetime
import sys
original_stdout = sys.stdout

pygame.init()
pygame.display.set_mode([200,200])
pygame.key.set_repeat()
ser = serial.Serial('COM12', 115200, timeout=1)
ser.flush()	
SPEED = 0
import sys

DIRECTION= 30
def writeArduiono(d, s):
    ACTION = (str(s)+"#" +str(d)+ "\n").encode('utf_8')
    ser.write(ACTION)
    line = ser.readline().decode('utf-8').rstrip()	
    # print(line)
#  main
pygame.key.set_repeat()
write = False
count = 1

f = open("filename.txt", 'w')
sys.stdout = f
while True: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); #sys.exit() if sys is imported
        if event.type == pygame.KEYDOWN:
            if event.key == ord('w'):
                SPEED = 6
            elif event.key == ord('s'):
                SPEED = -6
            if event.key == ord('a'):
                DIRECTION = 60
            elif event.key == ord('d'):
                DIRECTION = 0
            if event.key == ord('l'):
                if write == False:
                    write = True
                elif write == True:
                    write = False
            if event.key == ord('q'):
                sys.exit()

        if event.type == pygame.KEYUP:
            if event.key == ord('w') or event.key == ord('s'):
                SPEED = 0
            if event.key == ord('a') or event.key == ord('d'):
                DIRECTION = 30
    writeArduiono(DIRECTION, SPEED)
    if write == True:
        print(f"{SPEED}, {DIRECTION}")  

sys.exit()