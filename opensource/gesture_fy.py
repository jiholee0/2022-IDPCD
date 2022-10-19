# 출처 : https://velog.io/@dlth508/Toy-Project-%EA%B0%80%EC%9A%B4%EB%8D%B0-%EC%86%90%EA%B0%80%EB%9D%BD-%EB%AA%A8%EC%9E%90%EC%9D%B4%ED%81%AC-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%A7%8C%EB%93%A4%EA%B8%B0

import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 5	# 최대 몇 개의 손을 인식할 건지 정의
gesture = {		# fy(가운데 손가락) 클래스 정의 -> 11
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok', 11: 'fy'
}

# MediaPipe hands model (first model: 손 감지 모델)
# 영상에서 손가락 뼈 마디 부분(연두색 선, 빨간점)을 그릴 수 있도록 도와주는 것
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 손가락 detection 모듈 초기화
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,  # 최대 몇 개의 손을 인식할건지 정의
    min_detection_confidence=0.5, # 0.5로 해두는게 좋음
    min_tracking_confidence=0.5)

# Gesture recognition model (두번째 모델: 제스처 인식 모델)
# gesture_train_fy.csv: 손가락의 각도들과 마지막에 label 값 저장되어 있는 파일
file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',') # data file
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create() 		   # K-Nearest Neighbors 알고리즘(K-최근접 알고리즘)
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # 학습 시키기

cap = cv2.VideoCapture(0) # 웹캠의 이미지 읽어옴

while cap.isOpened():
    ret, img = cap.read() # 한 프레임씩 이미지 읽어옴
    if not ret:
        continue

    # 전처리 / opencv: BGR, mediapipe: RGB
    img = cv2.flip(img, 1) # 이미지가 거울에서 보는 거 처럼 뒤집어져 있기 때문에 좌우 반전 해줌
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 전처리 된 이미지
    result = hands.process(img) # 전처리 및 모델 추론을 함께 실행함

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 출력을 위해 다시 바꿔줌

    if result.multi_hand_landmarks is not None: # 만약 손을 인식했다면
        for res in result.multi_hand_landmarks: # 여러 개의 손을 인식할 수 있기 때문에 for문 사용
            joint = np.zeros((21, 3)) # joint -> 빨간점, 21개의 joint / 빨간점의 x, y, z 3개의 좌표이므로 3
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z] # 각 joint 마다 landmark 저장 (landmark의 x, y, z 좌표 저장)

            # Compute angles between joints (각 joint 번호의 인덱스 나와있음)
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :] # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :] # Child joint
            v = v2 - v1 # [20,3]

            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 길이로 나눠줌 (크키 1짜리 vector 나오게 됨(unit vector))

            # Get angle using arcos of dot product (15개의 각도 구하기)
            angle = np.arccos(np.einsum('   nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :])) # [15,]

            # Convert radian to degree
            angle = np.degrees(angle) # angle이 radian 값으로 나오기 때문에 degree 값으로 바꿔줌

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            # print(data)
            # data = np.append(data, 11)
            ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때의 값 구함
            idx = int(results[0][0])

            if idx == 11:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                # # 좌표가 잘 구해졌는데 사각형 그려보기
                # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=2)

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST) # 이미지 크기를 0.05배로 (작게) 만듬듬
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break
