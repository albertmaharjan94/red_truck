import datetime
import sys
import time
original_stdout = sys.stdout

f = open("time.txt", 'w')
sys.stdout = f

t_end = time.time() + 10
prev = round(time.time() * 1000)
while time.time() < t_end:
  dt = round(time.time() * 1000)
  if(prev !=dt):
    print(dt)
    prev = dt



