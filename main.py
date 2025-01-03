import cv2
import pyautogui
import time
import mediapipe as mp
from hand_recognition import HandRecognition
from gestures import are_fingers_spread, is_thumb_up

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

hand_recognition = HandRecognition()

last_click_time = 0
hover_start_time = 0
hovering = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    results = hand_recognition.process_frame(frame)

    if results.multi_hand_landmarks:
        hand_recognition.draw_landmarks(frame, results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            pyautogui.moveTo(x, y)

            is_spread = are_fingers_spread(hand_landmarks)
            thumb_up = is_thumb_up(hand_landmarks)

            current_time = time.time()

            if is_spread:
                if not hovering:
                    hover_start_time = current_time
                    hovering = True
                elif current_time - hover_start_time > 0.5:
                    if current_time - last_click_time > 0.3:
                        pyautogui.click()
                        last_click_time = current_time
            elif thumb_up:
                if current_time - last_click_time > 0.3:
                    pyautogui.rightClick()
                    last_click_time = current_time
            else:
                if hovering:
                    hovering = False

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
