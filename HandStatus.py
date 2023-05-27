import cv2

class Tracking:
    def __init__(self, mp_drawing, mp_drawing_styles, mp_hands, hands, print):
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.mp_hands = mp_hands
        self.hands = hands
        self.print = print
    
    def process(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        # Draw the hand annotations on the image.

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return (image, results)

    def get_hand_lr(self, results):
        hand_lr = results.multi_handedness
        hand_lr = str(hand_lr).split(':')[3].split('"')[1]
        if hand_lr == 'Left':
            hand_lr = 'Right'
        elif hand_lr == 'Right':
            hand_lr = 'Left'
        
        if self.print:
            print(hand_lr)
        return hand_lr
    
    def get_fingers_status(self, image, image_height, hand_landmarks, results, draw):
        # 5개 손가락의 마디 위치 관계를 확인하여 플래그 변수를 설정합니다. 손가락을 일자로 편 상태인지 확인합니다.
        thumb_finger_state = 0
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].y * image_height:
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y * image_height:
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                    thumb_finger_state = 1

        index_finger_state = 0
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                    index_finger_state = 1

        middle_finger_state = 0
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                    middle_finger_state = 1

        ring_finger_state = 0
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                    ring_finger_state = 1

        pinky_finger_state = 0
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y * image_height:
            if hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y * image_height:
                if hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y * image_height > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y * image_height:
                    pinky_finger_state = 1
        
        fingers_staus = {
            'thumb':thumb_finger_state,
            'index':index_finger_state,
            'middle':middle_finger_state,
            'ring':ring_finger_state,
            'pinky':pinky_finger_state
            }

        if draw:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if self.print:
            print(fingers_staus)
        return fingers_staus
    
    def get_hand_pos(self, hand_landmarks):
        hand_pos = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        hand_pos.x = 1 - hand_pos.x
        hand_pos.y = 1 - hand_pos.y
        hand_pos.z = 1 - hand_pos.z

        return hand_pos