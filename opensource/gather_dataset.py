import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1 # 인식할 수 있는 손 개수
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
} # 12가지의 제스처, 제스처 데이터는 손가락 관절의 각도와 각각의 라벨을 뜻한다.

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition data
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
print(file.shape)

cap = cv2.VideoCapture(0)

def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            data = np.array([angle], dtype=np.float32)
            
            data = np.append(data, 11) # 

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img) 
    if cv2.waitKey(1) == ord('q'):
        break

np.savetxt('data/gesture_train_fy.csv', file, delimiter=',')

#