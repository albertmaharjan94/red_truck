import serial
import time

class Arduino:
    def __init__(self, dev = '/dev/ttyUSB0'):        
        self.ser = serial.Serial(dev, 250000, timeout=1.5)
        time.sleep(3)
        self.ser.flush()

    
    def arduino_process(self, _running, _speed, _direction):
        while _running.value:
            print(f"speed {_speed.value} direction {_direction.value}")
            ACTION = (str(_speed.value)+"#" +str(_direction.value)+ "\n").encode('utf_8')
            self.ser.write(ACTION)
            try:
                line = self.ser.readline().decode('utf-8').rstrip()	
                print(line)
            except Exception as e:
                print(e)