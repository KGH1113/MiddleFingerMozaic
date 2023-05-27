import cv2
import mediapipe as mp
from HandStatus import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
CarStatus = None

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    tracking = Tracking(mp_drawing, mp_drawing_styles, mp_hands, hands, False)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignored camera frame!")
            continue
        
        image, results = tracking.process(image)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = [float(str(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x)), float(str(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y))]
                width = [float(str(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)) - x[0], float(str(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)) - x[1]]
                y = [float(str(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)), float(str(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y))]
                height = [float(str(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)) - y[0], float(str(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)) - y[1]]
                
                x[0] = x[0] * image_width
                x[1] = x[1] * image_height
                y[0] = y[0] * image_width
                y[1] = y[1] * image_height

                print(x)
                print(width)
                print(y)
                print(height)

                finger_status = tracking.get_fingers_status(image, image_height, hand_landmarks, results, draw=True)
                hand_pos = tracking.get_hand_pos(hand_landmarks)
            hand_lr = tracking.get_hand_lr(results)

            if (
                # finger_status['thumb'] == 0 and \   # Reason that this code is comment: There is an error that status of thumb always returns 0
                finger_status['index'] == 0 and \
                finger_status['middle'] == 1 and \
                finger_status['ring'] == 0 and \
                finger_status['pinky'] == 0
                ):
                image = mosaic_area(image, x, y, width, height)

        cv2.imshow('Middle_Mozaic', cv2.flip(image, 1))
        # Flip the image horizontally for a selfie-view display.
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
