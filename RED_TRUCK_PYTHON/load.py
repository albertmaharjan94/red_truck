import time
import serial
import csv
import datetime
import sys
import threading

import time 
ser = serial.Serial('COM12', 115200, timeout=1)

ser.flush()	
time.sleep(3)
SPEED = 0
DIRECTION = 260
TIME_BIAS =6
running = True
f =  open('filename.txt')
lines = f.read().splitlines()
# while running:
for i in lines[::TIME_BIAS]:
    _split = i.split(', ')
    SPEED = _split[0]
    DIRECTION = _split[1]
    time.sleep(0.0002)
    ACTION = (str(SPEED)+"#" +str(DIRECTION)+ "\n").encode('utf_8')
    ser.write(ACTION)
    line = ser.readline().decode('utf-8').rstrip()	
    print(line)
print("FINISH")