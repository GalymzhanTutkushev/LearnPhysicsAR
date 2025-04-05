# import the necessary packages

from imutils.video.videostream import VideoStream
import time

import time
import cv2
import numpy as np
 
# initialize the camera and grab a reference to the raw camera capture
camera = VideoStream().start()
images=[]
# allow the camera to warmup
time.sleep(2.0)
C = 961.302
k = 1.045
avg_distance = 100
alpha = 0.1
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
#	image=frame.array
# capture frames from the camera

	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	
while True:
    image = camera.read()
    #image = image[100:150,:]
    r = 640.0 / image.shape[1]
    dim = (640, int(image.shape[0] * r))
 
# perform the actual resizing of the image and show it
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
    gray = cv2.medianBlur(gray,5)
    rows = gray.shape[0]
    upper_text = ""
    font = cv2.FONT_HERSHEY_DUPLEX
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/8,param1=100,param2=30,minRadius=5,maxRadius=45)
    if circles is not None:
        radius = circles[0][0][2]
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            cv2.circle(image, (x,y), r, (0, 255, 0), 4)
            print(x,y,r)
            if radius > 0.0:
                distance = C / np.power(radius, k)
		#cur_dist = alpha * distance + (1-alpha) * avg_distance
		#avg_distance = cur_dist
		#upper_text = str(avg_distance)
	# show the frame
#    cv2.putText(image, upper_text, (30, 30), font, 0.6, (255, 255, 255), 1)
    cv2.imshow("Frame", image)
		# clear the stream in preparation for the next frame
    #rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
	
 

cv2.destroyAllWindows()
