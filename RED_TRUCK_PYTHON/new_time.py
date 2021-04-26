from time import time, sleep
while True:
    sleep(60 - time() % 60)
    print("test")