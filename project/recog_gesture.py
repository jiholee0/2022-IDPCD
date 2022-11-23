import cv2 # 웹캠 제어 및 ML 사용
import mediapipe as mp # 손 인식을 할 것
import numpy as np
import time
from scipy import stats

def recog_gesture() -> str :
    result_list = []
    # 제스처 인식
    max_num_hands = 1 # 손은 최대 1개만 인식
    gesture = { # **11가지나 되는 제스처 라벨, 각 라벨의 제스처 데이터는 이미 수집됨 (제스처 데이터 == 손가락 관절의  각도, 각각의 라벨)**
        0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
        6:'six', 7:'rock', 8:'spiderman', 9:'scissors', 10:'ok'
    }
    kiosk_gesture = {0:'zero', 1:'one', 2:'two', 5:'five', 10:'ok'} # 우리가 사용할 제스처 라벨만 가져옴

    # MediaPipe hands model
    mp_hands = mp.solutions.hands # 웹캠 영상에서 손가락 마디와 포인트를 그릴 수 있게 도와주는 유틸리티1
    mp_drawing = mp.solutions.drawing_utils # 웹캠 영상에서 손가락 마디와 포인트를 그릴 수 있게 도와주는 유틸리티2

    # 손가락 detection 모듈을 초기화
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands, # 최대 몇 개의 손을 인식?
        min_detection_confidence=0.5, # 0.5로 해두는 게 좋다!
        min_tracking_confidence=0.5)

    # 제스처 인식 모델
    file = np.genfromtxt('C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/data/gesture_train.csv', delimiter=',') # 각 제스처들의 라벨과 각도가 저장되어 있음, 정확도를 높이고 싶으면 데이터를 추가해보자!**
    angle = file[:,:-1].astype(np.float32) # 각도
    label = file[:, -1].astype(np.float32) # 라벨
    knn = cv2.ml.KNearest_create() # knn(k-최근접 알고리즘)으로
    knn.train(angle, cv2.ml.ROW_SAMPLE, label) # 학습!

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cap.isOpened():
        number = 0
        target_tick = time.time() + 2

        while time.time() < target_tick :
            number += 1
            ret, img = cap.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 각도를 인식하고 제스처를 인식하는 부분
            if result.multi_hand_landmarks is not None: # 만약 손을 인식하면
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3)) # joint == 랜드마크에서 빨간 점, joint는 21개가 있고 x,y,z 좌표니까 21,3
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z] # 각 joint마다 x,y,z 좌표 저장

                    # Compute angles between joints joint마다 각도 계산
                    # **공식문서 들어가보면 각 joint 번호의 인덱스가 나옴**
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]관절벡터
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 벡터 정규화(크기 1 벡터) = v / 벡터의 크기

                    # Get angle using arcos of dot product **내적 후 arcos으로 각도를 구해줌**
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    # Inference gesture 학습시킨 제스처 모델에 참조를 한다.
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때 값을 구한다!
                    idx = int(results[0][0]) # 인덱스를 저장!

                    # Draw gesture result
                    if idx in kiosk_gesture.keys():
                        cv2.putText(img, text=kiosk_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        rst = kiosk_gesture[idx].upper()

                    # Other gestures 모든 제스처를 표시한다면
                    # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손에 랜드마크를 그려줌

            resize = cv2.resize(img, (200, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('gesture', resize)
            if cv2.waitKey(1) == ord('q'):
                break
    print('--------------\n total number : ', number)
    print('result list : ', result_list, '\n')
    mode = stats.mode(result_list)[0]
    if len(mode) == 0 : return 'fail'
    else : return mode[0]

# print(recog_gesture())