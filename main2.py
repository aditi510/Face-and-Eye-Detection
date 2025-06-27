import cv2
import time
import numpy as np
import winsound
from collections import deque
from datetime import datetime

face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

eye_closed_start = None
alert_level = 0
font = cv2.FONT_HERSHEY_SIMPLEX

alert_sounds = {
    1: (1000, 300), 
    2: (1500, 500) 
}

log_file = open("drowsiness_log.txt", "a")

blink_count = 0
blink_start = time.time()
prev_eye_state = 'Open'

frame_times = deque(maxlen=30)  

def is_eye_open(eye_gray):
    local_image = cv2.blur(eye_gray, (5, 5))
    _, thresh = cv2.threshold(local_image, 132, 255, cv2.THRESH_BINARY)

    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    white_ratio = white_pixels / total_pixels

    return white_ratio > 0.2

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.1, 5)

    eye_state = 'Not Detected'
    current_time = time.time()
    eye_checked = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cas.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]

            if is_eye_open(eye_gray):
                eye_state = 'Open'
                alert_level = 0
                eye_closed_start = None

                if prev_eye_state == 'Closed':
                    blink_count += 1

            else:
                eye_state = 'Closed'
                if eye_closed_start is None:
                    eye_closed_start = current_time
                else:
                    closed_duration = current_time - eye_closed_start
                    if closed_duration < 0.2:
                        alert_level = 0
                    elif 0.2 <= closed_duration <= 2:
                        if alert_level != 1:
                            winsound.Beep(*alert_sounds[1])
                            log_file.write(f"[{datetime.now()}] Level 1: Slightly Drowsy\n")
                        alert_level = 1
                    else:
                        if alert_level != 2:
                            winsound.Beep(*alert_sounds[2])
                            log_file.write(f"[{datetime.now()}] Level 2: Drowsiness Detected\n")
                        alert_level = 2
            prev_eye_state = eye_state
            eye_checked = True

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            break
        break

    if not eye_checked:
        eye_state = 'No Eye Found'

    if alert_level == 1:
        cv2.putText(frame, "Stay Alert!", (30, 60), font, 1, (0, 255, 255), 2)
    elif alert_level == 2:
        cv2.putText(frame, "DROWSINESS DETECTED!", (30, 60), font, 1, (0, 0, 255), 3)

    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0

    elapsed_time = time.time() - blink_start
    if elapsed_time > 10: 
        blink_rate = (blink_count / elapsed_time) * 60 
        blink_start = time.time()
        blink_count = 0
    else:
        blink_rate = (blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0

    cv2.putText(frame, f'Eye State: {eye_state}', (30, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Alert Level: {alert_level}', (30, 90), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Blink Rate: {blink_rate:.1f} blinks/min', (30, 120), font, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f'FPS: {fps:.1f}', (30, 150), font, 0.6, (200, 200, 200), 1)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()