import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandRecognition:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7):
        self.h nds = mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, hand_landmarks):
        if hand_landmarks:
            for landmarks in hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
