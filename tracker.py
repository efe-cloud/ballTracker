import cv2
import numpy as np
import os


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


# Load settings
settings_file = 'slider_values.txt'
settings = read_settings(settings_file)

# settings window
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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Update settings
        hl = cv2.getTrackbarPos('Hue Lower', 'settings')
        hu = cv2.getTrackbarPos('Hue Upper', 'settings')
        sl = cv2.getTrackbarPos('Sat Lower', 'settings')
        su = cv2.getTrackbarPos('Sat Upper', 'settings')
        vl = cv2.getTrackbarPos('Val Lower', 'settings')
        vu = cv2.getTrackbarPos('Val Upper', 'settings')
        recording = cv2.getTrackbarPos('Trajectory', 'settings')

        lower_yellow = np.array([hl, sl, vl])
        upper_yellow = np.array([hu, su, vu])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                if recording:  # Only record when the toggle is 'On'
                    trajectory.append((int(x), int(y)))

                # Draw the trajectory
                for i in range(1, len(trajectory)):
                    if trajectory[i - 1] is None or trajectory[i] is None:
                        continue
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', cv2.bitwise_and(frame, frame, mask=mask))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    # Save settings
    current_settings = {
        'Hue Lower': hl, 'Hue Upper': hu,
        'Sat Lower': sl, 'Sat Upper': su,
        'Val Lower': vl, 'Val Upper': vu,
        'Trajectory': recording
    }
    save_settings(settings_file, current_settings)
