from __future__ import print_function
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import imutils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
socketio = SocketIO(app, async_mode='threading')

pin1 = 2
pin2 = 3
GPIO.setwarnings(False)  # Turn off warnings output
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin1, GPIO.OUT, initial=0)
GPIO.setup(pin2, GPIO.OUT, initial=1)
# initialize the camera and grab a reference to the raw camera capture
frames = 120
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
# allow the camera to warmup
avg_acceleration = []
avg_friction = []


@app.route('/')
def main():
    return render_template('main.html')

@socketio.on('restart')
def restart_experiment():
    GPIO.setup(pin1, GPIO.OUT, initial=0)
    GPIO.setup(pin2, GPIO.OUT, initial=1)

@socketio.on('start')
def start_experiment():
    GPIO.output(pin1, GPIO.LOW)
    GPIO.output(pin2, GPIO.LOW)
    t = []
    images = []
    #st = time.time()
    time.sleep(0.1)
    t_start = time.time()
    for i in range(0,frames):
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        images.append(frame)
        t.append(time.time())
        # update the FPS counter
        #fps.update()

    t_end = time.time()
    #fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("Elapsed time: {:.2f}".format(t_end - t_start))
    myfps = (1+frames) / (t_end - t_start)
    print("My FPS: {:.2f}".format(myfps))

    C = 961.302
    k = 1.045
    distances = []
    ts=[]
    
    # capture frames from the camera
    for i in range(0,frames):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text

        image = images[i][60:260, 200:400]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=100)
        if circles is not None:
            radius = circles[0][0][2]
            if radius > 0.0:
                distance = C / np.power(radius, k)
                distances.append(74 - distance)
                ts.append(t[i])
        
            
        
            

#    vs.stop()
    
    t0 = t[0]
    data_to_clear = np.asarray(distances)
    time_to_clear = np.asarray(ts) - t0
    data_to_clear = np.insert(data_to_clear,0,0)
    time_to_clear = np.insert(time_to_clear,0,0)
    t_buf = [time_to_clear[0]]
    x_buf = [data_to_clear[0]]
    x_val_last = data_to_clear[0]
    time_to_fit = []
    data_to_fit = []
    for i in range(1,len(data_to_clear)-1):
        if abs(x_val_last - data_to_clear[i]) < 0.1:
            x_val = data_to_clear[i]
            x_buf.append(data_to_clear[i])
            t_buf.append(time_to_clear[i])
        else:
            t_cur = np.asarray(t_buf)
            x_cur = np.asarray(x_buf)
            time_to_fit.append(t_cur.mean())
            data_to_fit.append(x_cur.mean())
            x_buf = [data_to_clear[i]]
            t_buf = [time_to_clear[i]]
            x_val_last = data_to_clear[i]

    print(len(time_to_fit), len(data_to_fit))
    time_to_fit = np.asarray(time_to_fit)
    data_to_fit = np.asarray(data_to_fit)
        
    p = np.polyfit(time_to_fit, data_to_fit, 2)

    global avg_acceleration
    global avg_friction
    sina = 0.485
    cosa = np.sqrt(1-sina*sina)
    g = 9.81
    aex = 2*p[0] / 100
    ath = g*sina
    frc = (ath-aex)/(g*cosa)
    avg_acceleration.append(aex)
    avg_friction.append(frc)
    print("time :" + str(len(time_to_fit)))
    print("data :" + str(len(data_to_fit)))
    print("acceleration: " + str(aex))
    print("acceleration theor: " + str(ath))
    print("friction coeff: " + str(frc))

    data_array = []
    t_last = 0
    for i in range(0, len(data_to_fit)):
        data_chunk = [time_to_fit[i], data_to_fit[i]]
        t_last = time_to_fit[i]
        data_array.append(data_chunk)
    
    socketio.emit('data_delivery_event', data_array)
    
    fit_array = []
    for t_local in np.linspace(0, t_last, 100):
        fit_chunk = [t_local, p[2] + p[1] * t_local + p[0] * t_local * t_local]
        fit_array.append(fit_chunk)
    socketio.emit('fit_delivery_event', fit_array)
    np_avg_acc = np.asarray(avg_acceleration)
    np_avg_frc = np.asarray(avg_friction)
    exp_data = [aex, frc, np_avg_acc.mean(), np_avg_frc.mean(), np_avg_acc.std(), np_avg_frc.std()]
    print(exp_data)
    socketio.emit('plot_event', exp_data)

if __name__ == '__main__':
    socketio.run(app, debug=False)
