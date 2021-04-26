import sys
import threading
import time 

running = True
f =  open('time.txt')
lines = f.read().splitlines()
lines = list(set(lines))

input("Ready?")
for i in lines[::6]:
    current = round((time.time()*1000)%10)
    _split = i.split(', ')
    fm= int(_split[0])%10
    while(fm!= current):
        current = round((time.time()*1000)%10)
    print(f"{fm}  {_split[0]}  {current}") 
print("FINISH")
