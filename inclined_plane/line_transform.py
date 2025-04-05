import cv2
import numpy as np
from imutils.video.pivideostream import PiVideoStream
import time

camera = PiVideoStream().start()
time.sleep(2.0)


while True:
    frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,100, min_theta = 0, max_theta = 2 * np.pi/180)
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('lineTransform',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()