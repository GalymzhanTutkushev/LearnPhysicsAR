from imutils.video.pivideostream import PiVideoStream
import imutils
import time
import cv2
import numpy as np


class CircleTracker:
    def __init__(self):
        self.camera = PiVideoStream().start()
        time.sleep(2.0)        
        self.min_radius = 15
        self.max_radius = 55
        self.single_circle_system = True
        self.crop_x = 0
        self.crop_y = 0
        self.circles_in_frame = []
        self.autofocus()
        
    def autofocus(self):
        crop_regions = [[0, 0],
                        [80, 0],
                        [160, 0],
                        [240, 0],
                        [0, 80],
                        [80, 80],
                        [160, 80],
                        [240, 80],
                        [0, 160],
                        [80, 160],
                        [160, 160],
                        [240, 160]]
        x, y = 0, 0        
        for ROI in crop_regions:
            frame = self.camera.read()
            frame = imutils.resize(frame[ROI[1]:ROI[1] + 80, ROI[0]:ROI[0] + 80], width=300)
            self.track_circles(frame)            
            if self.circles_in_frame is not None:
                if self.circles_in_frame[0][2] > 0:
                    x = ROI[0]
                    y = ROI[1]                    
        x_array = []
        y_array = []
        circs = 0
        while circs < 10:
            frame = self.camera.read()
            frame = imutils.resize(frame[int(y):int(y) + 80, int(x):int(x) + 80], width=300)
            self.track_circles(frame)            
            if self.circles_in_frame is not None:
                x_array.append(self.circles_in_frame[0][0])
                y_array.append(self.circles_in_frame[0][1])
                circs += 1
        self.crop_x = int(x+4*np.mean(np.asarray(x_array))/15-40)
        self.crop_y = int(y+4*np.mean(np.asarray(y_array))/15-40)
            
        
    def get_frame(self):
        frame = self.camera.read()
        frame = imutils.resize(frame[self.crop_y:self.crop_y + 80, self.crop_x:self.crop_x + 80], width=300)
        return frame
    
    def track_circles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30,
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        if circles is not None:
            for all_circles in circles:
                self.circles_in_frame = all_circles
            if self.single_circle_system:
                self.circles_in_frame = [self.circles_in_frame[0]]
        else:
            self.circles_in_frame = None
    
    def __del__(self):
        self.camera.stop()
