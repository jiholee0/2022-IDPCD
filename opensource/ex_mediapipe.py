# 출처 : https://makernambo.com/154 https://blog.naver.com/PostView.naver?blogId=skyjjw79&logNo=222327014865&parentCategoryNo=54&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
# mediapipe 모듈을 import하고 약식으로 사용할 명칭을 지정한다.
# mediapipe.solutions.hands 모듈이 손동작인식을 위한 모듈이다. 
import cv2
import mediapipe as mp
 
mp_drawing = mp.solutions.drawing_utils # 손 위에 그림을 그릴 수 있는 메소드
mp_hands = mp.solutions.hands
 
# OpenCV로 웹캠을 읽어 입력데이터 소스로 지정한다. 
cap = cv2.VideoCapture(0) # USB 캠을 활용하고 싶다면 0 대신 1
 
# hands 손가락 인식모듈의 작동 option을 지정한다. 
with mp_hands.Hands(    
    max_num_hands=1, # 인식할 손모양의 갯수, 생략하면 2가 지정된다.
    min_detection_confidence=0.5, # 성공적인 것으로 간주되는 최소 신뢰도 값. 0.0 ~1.0사이로서 기본값은 0.5이다.
    min_tracking_confidence=0.5) as hands: # 손 랜드마크가 성공적으로 추적된 것으로 간주되는 최소 신뢰도 값. 0.0 ~1.0 사이로서 기본값은 0.5이다. 이 값을 높이면 시간이 더 소요되지만 좀 더 정확한 작동이 보장된다. 
 
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        # OpenCV 영상은 BGR 형식인데 MediaPipe에서는 RGB 형식을 사용하므로 영상형식을 변환해 준다 
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
 
        # MediaPipe의 hands 모듈을 이용해서 손동작을 인식한다. 손동작 인식 AI모델이 작동되고 결과 값이 result로 저장된다. 
        results = hands.process(image)
        
        # MediaPipe용 RGB 형식으로 변환했던 것을 OpenCV 영상저리를위해 다시 BGR형식으로 되돌린다. 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # result값이 정상인 경우에만 후속 작업 처리한다. 
        if results.multi_hand_landmarks: # 실제 손이 인식될 때 점(랜드마크)을 찍는다.
            # result로 반환된 landmark 데이터를 사용한다. 인식된 손가락 모양은 index값을 가지는 배열로 제공된다.
            for hand_landmarks in results.multi_hand_landmarks:
                finger1 = int(hand_landmarks.landmark[4].x * 100 ) # 엄지손가락 끝의 X좌표를 백분율로 표시한 것.
                finger2 = int(hand_landmarks.landmark[8].x * 100 ) # 검지손가락 끝의 X좌표를 백분율로 표시한 것.
                
                # 두 손가락 끝의 x좌표값 차이의 절대값을 구해 두 손가락 끝이 벌어진 정도를 계산한다. 
                dist = abs(finger1 - finger2)
                cv2.putText(
                    image, text='f1=%d f2=%d dist=%d ' % (finger1,finger2,dist), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
 
                # MediaPipe에 내장된 유틸리티 기능을 이용해서 구해진 손가락 모양을 서로 연결한 그림을 그려준다.
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS) # 연결선
 
        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break
 
cap.release()
