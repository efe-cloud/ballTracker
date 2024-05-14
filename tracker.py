import cv2
import numpy as np
import os
import time

def nothing(x):
    pass

def read_settings(filename):
    default_settings = {
        'Hue Lower': 25, 'Hue Upper': 35,
        'Sat Lower': 100, 'Sat Upper': 255,
        'Val Lower': 100, 'Val Upper': 255,
        'Trajectory': 0
    }
    if not os.path.exists(filename):
        return default_settings

    with open(filename, 'r') as file:
        lines = file.readlines()
        settings = {}
        for line in lines:
            key, value = line.strip().split(': ')
            settings[key] = int(value)
        return settings

def save_settings(filename, settings):
    with open(filename, 'w') as file:
        for key, value in settings.items():
            file.write(f"{key}: {value}\n")

def predict_future_position(trajectory, seconds_ahead):
    if len(trajectory) < 2:
        return None  # not enough data

    (x1, y1), (x2, y2) = trajectory[-2:]
    vx = (x2 - x1)
    vy = (y2 - y1)

    future_x = x2 + vx * seconds_ahead
    future_y = y2 + vy * seconds_ahead
    return (int(future_x), int(future_y))

settings_file = 'slider_values.txt'
settings = read_settings(settings_file)

cv2.namedWindow('settings')
cv2.createTrackbar('Hue Lower', 'settings', settings['Hue Lower'], 180, nothing)
cv2.createTrackbar('Hue Upper', 'settings', settings['Hue Upper'], 180, nothing)
cv2.createTrackbar('Sat Lower', 'settings', settings['Sat Lower'], 255, nothing)
cv2.createTrackbar('Sat Upper', 'settings', settings['Sat Upper'], 255, nothing)
cv2.createTrackbar('Val Lower', 'settings', settings['Val Lower'], 255, nothing)
cv2.createTrackbar('Val Upper', 'settings', settings['Val Upper'], 255, nothing)
cv2.createTrackbar('Trajectory', 'settings', settings['Trajectory'], 1, nothing)

cap = cv2.VideoCapture(0)
trajectory = []

try:
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror the frame

        # apply gaussian blur reduce noise
        blurred_frame = cv2.GaussianBlur(frame, (11, 11), 0)

        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        hl, hu, sl, su, vl, vu = [cv2.getTrackbarPos(name, 'settings') for name in
                                  ['Hue Lower', 'Hue Upper', 'Sat Lower', 'Sat Upper', 'Val Lower', 'Val Upper']]
        lower_yellow = np.array([hl, sl, vl])
        upper_yellow = np.array([hu, su, vu])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = [c for c in contours if cv2.contourArea(c) > 100]
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    trajectory.append((int(x), int(y)))

                    # draw the trajectory
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)  # Green lines for the trajectory

                    # dodge instruction if half of the screen passed
                    if y > frame.shape[0] / 2:
                        future_position = predict_future_position(trajectory, 5)
                        if future_position:
                            cv2.circle(frame, future_position, 10, (0, 0, 255), 3)
                            cv2.arrowedLine(frame, (int(x), int(y)), future_position, (255, 0, 0), 2)

                            # dodge decision based on future position
                            if future_position[0] < frame.shape[1] // 2:
                                print("right")
                            else:
                                print("left")

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', cv2.bitwise_and(frame, frame, mask=mask))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    current_settings = {
        'Hue Lower': hl, 'Hue Upper': hu,
        'Sat Lower': sl, 'Sat Upper': su,
        'Val Lower': vl, 'Val Upper': vu,
        'Trajectory': cv2.getTrackbarPos('Trajectory', 'settings')
    }
    save_settings(settings_file, current_settings)
