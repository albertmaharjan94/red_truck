import socket
import time
import io
import base64
from imageio import imread
import cv2

# socket init
HOST = '172.25.0.48'
PORT = 65432   

while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        data = s.recv(1000000) # receive bytes in 1000000 buffer for this frame
        data = data.decode("utf-8")
        _split = data.split("#")
        img = imread(io.BytesIO(base64.b64decode(_split[0])))
        print(_split[1])
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        t = cv2.waitKey(1)
        if t == ord('q'):
            s.close()
            break
        cv2.imshow("client",cv2_img)    
