from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
import numpy as np
from scipy.optimize import leastsq
from circle_tracker import CircleTracker

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
socketio = SocketIO(app, async_mode='threading')

cx_data = []  # array of x coordinates to calculate motion parameters
cy_data = []  # array of y coordinates to calculate motion parameters
t_data = []  # array of time points where motion parameters have been captured
dx_data = []  # array of x coordinates to display
dy_data = []  # array of y coordinates to display
img = 255*np.ones(shape=(900, 900, 3), dtype=np.uint8)  # final image to output in the client
img_x_canvas_area = 255*np.ones(shape=(600, 300, 3), dtype=np.uint8)  # white vertical canvas area
img_y_canvas_area = 255*np.ones(shape=(300, 600, 3), dtype=np.uint8)  # white horizontal canvas area

t_start = 0
t_end = 0
pen_length = 1  # pendulum length (to be set from the client)
pen_length_changed = False
start_capture = False  # START signal
stop_capture = False  # STOP signal
plot_area_length = 570  # pixel size of the plotting area in the client
max_array_length = 50  # maximum length of the array to plot
delta = plot_area_length / max_array_length  # delta step to plot data
start_plot = 333  # starting point to plot data in pixels


@app.route('/')
def main():
    return render_template('main.html')


def fit_to_sine(x, y):  # fit to the y(x) = A0 + A1 * sin(A2*x + A3) function
    guess_mean = np.mean(y)  # guess the mean value (A0)
    guess_amp = (np.max(y) - np.min(y)) / 2  # guess the amplitude (A1)
    guess_freq = np.sqrt(981 / pen_length)  # guess the frequency (A2)
    guess_phase = 0  # guess the initial phase (A3)
    optimize_func = lambda p: p[0]*np.sin(p[1]*x+p[2]) + p[3] - y
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]    
    theta = (y-est_mean)/est_amp
    for i in range(0, len(theta)-1):
        if theta[i] > 1.0:
            theta[i] = 1.0
        elif theta[i] < -1.0:
            theta[i] = -1.0    
    guess_phase = np.mean(np.arcsin(theta)-est_freq*x)    
    print(est_amp, est_freq, est_phase, est_mean)
    return est_mean, est_amp, est_freq, guess_phase


def draw_canvas_lines():
    global img
    img[300:900, 0:300] = img_x_canvas_area
    img[0:300, 300:900] = img_y_canvas_area
    cv2.line(img, (0, 0), (0, 900), (0, 0, 0), 1)
    cv2.line(img, (300, 0), (300, 900), (0, 0, 0), 1)
    cv2.line(img, (0, 0), (900, 0), (0, 0, 0), 1)
    cv2.line(img, (0, 300), (900, 300), (0, 0, 0), 1)
    cv2.line(img, (0, 899), (300, 899), (0, 0, 0), 1)
    cv2.line(img, (899, 0), (899, 300), (0, 0, 0), 1)


def draw_motion_points(x_array, y_array):
    global img
    if len(x_array) > 0:
        for i in range(0, len(x_array)-1):
            idx = len(x_array) - i - 1
            if x_array[idx] > -1:
                cv2.circle(img, (int(x_array[idx]), int(start_plot + i * delta)), 3, (74, 62, 235), 2)
                cv2.circle(img, (int(start_plot + i * delta), int(y_array[idx])), 3, (74, 62, 235), 2)
        idx = len(x_array) - 1
        if x_array[idx] > -1:
            cv2.line(img, (int(x_array[idx]), 0), (int(x_array[idx]), 330), (74, 62, 235), 2)
            cv2.line(img, (0, int(y_array[idx])), (330, int(y_array[idx])), (74, 62, 235), 2)
            cv2.circle(img, (int(x_array[idx]), start_plot), 3, (74, 62, 235), 2)
            cv2.circle(img, (start_plot, int(y_array[idx])), 3, (74, 62, 235), 2)


def draw_fitting_lines(x_mean, x_ampl, x_freq, x_phse, y_mean, y_ampl, y_freq, y_phse, t0, t1):
    global img
    time_array_to_fit = np.linspace(t0, t1, 200)
    coeff = (plot_area_length - delta) / (t1 - t0)
    xx_prev = x_mean + x_ampl * np.sin(x_freq * time_array_to_fit[len(time_array_to_fit)-1] + x_phse)
    yx_prev = y_mean + y_ampl * np.sin(y_freq * time_array_to_fit[len(time_array_to_fit)-1] + y_phse)
    dx_prev = start_plot + coeff * (time_array_to_fit[len(time_array_to_fit)-1] - t0)
    for i in range(1, len(time_array_to_fit)-1):
        xx_next = x_mean + x_ampl * np.sin(x_freq * time_array_to_fit[len(time_array_to_fit)-i-1] + x_phse)
        yx_next = y_mean + y_ampl * np.sin(y_freq * time_array_to_fit[len(time_array_to_fit)-i-1] + y_phse)
        dx_next = start_plot + coeff * (time_array_to_fit[len(time_array_to_fit)-i-1] - t0)
        cv2.line(img, (int(xx_prev), int(dx_prev)), (int(xx_next), int(dx_next)), (218, 190, 33), 2)
        cv2.line(img, (int(dx_prev), int(yx_prev)), (int(dx_next), int(yx_next)), (218, 190, 33), 2)
        xx_prev = xx_next
        yx_prev = yx_next
        dx_prev = dx_next


def gen(tracker):
    global t_data, cx_data, cy_data, start_capture, stop_capture, img, t_start, dx_data, dy_data, pen_length_changed
    while True:
        cstime = time.time()
        if pen_length_changed:
            mid_r = 955 / pen_length
            tracker.min_radius = (int)(mid_r-5)
            tracker.max_radius = (int)(mid_r+5)            
            pen_length_changed = False
            tracker.autofocus()
        frame = tracker.get_frame()  # get already pre-scaled current frame
        img[0:300, 0:300] = frame  # put the frame into image container top left side
        if start_capture:  # if the START button was pressed
            draw_canvas_lines()  # drawing canvas lines around image frame
            dx, dy = -1, -1
            tracker.track_circles(frame)  # track visible circles (single circle in this program)
            if tracker.circles_in_frame is not None:  # if a circle was found
                cur_time = time.time()  # fix current time
                if t_start == 0:  # if current time is the starting time, then fix it
                    t_start = cur_time
                cur_time = cur_time - t_start  # from now on, cur_time shows the time elapsed since the START event
                for circle in tracker.circles_in_frame:  # for each circle found (only one circle in this program)
                    cx = float(circle[0])  # x - coordinate
                    cy = float(circle[1])  # y - coordinate
                    dx, dy = cx, cy                    
                    cx_data.append(cx)  # collect x - coordinate in the cx_data Python array
                    cy_data.append(cy)  # collect y - coordinate in the cy_data Python array
                    t_data.append(cur_time)  # collect cur_time in the t_data Python array
            dx_data.append(dx)
            dy_data.append(dy)
            if len(dx_data) > max_array_length:  # shift all arrays to make their length not greater than maximum
                dx_data = dx_data[1:]
                dy_data = dy_data[1:]
                cx_data = cx_data[1:]
                cy_data = cy_data[1:]
                t_data = t_data[1:]
            draw_motion_points(dx_data, dy_data)

        if stop_capture:  # STOP button is pressed
            t = np.asarray(t_data)  # turn usual Python arrays into Numpy arrays
            x_array = np.asarray(cx_data)
            y_array = np.asarray(cy_data)
            x_mean, x_amp, x_freq, x_phase = fit_to_sine(t, x_array)  # fit x coordinates to sine function
            y_mean, y_amp, y_freq, y_phase = fit_to_sine(t, y_array)  # fit y coordinates to sine function
            draw_canvas_lines()
            draw_motion_points(dx_data, dy_data)
            #draw_fitting_lines(x_mean, x_amp, x_freq, x_phase, y_mean, y_amp, y_freq, y_phase, t[0], t[len(t)-1])

            freq = (x_freq + y_freq) / 2
            amp = np.sqrt(x_amp*x_amp + y_amp*y_amp)
            socketio.emit('exp_values', [2*np.pi / freq, amp])
            stop_capture = False
        
        ret, frame = cv2.imencode('.jpg', img)
        frame = frame.tobytes()
        #print(time.time() - cstime)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(CircleTracker()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('start_exp')
def start_experiment():
    global start_capture
    start_capture = True


@socketio.on('stop_exp')
def start_experiment():
    global start_capture, stop_capture
    start_capture = False
    stop_capture = True


@socketio.on('pen_length')
def pendulum_length_changes(l):
    global pen_length, pen_length_changed
    pen_length_changed = True
    pen_length = float(l)
    

if __name__ == '__main__':
    draw_canvas_lines()
    socketio.run(app, debug=False)
