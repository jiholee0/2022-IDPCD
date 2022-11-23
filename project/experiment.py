import sys
sys.path.append("/recog_gesture")
import recog_gesture as recog_gesture
import time

kiosk_gesture = ['ZERO', 'ONE', 'TWO', 'FIVE', 'OK']

def compare(count) -> str :
    isCorrect = 0
    isFail = 0
    for i in range(0,count) :
        for gesture in kiosk_gesture :
            result = recog_gesture.recog_gesture()
            print('return value : ', result , ' ', time.time(), '\n--------------\n')
            if result == gesture : isCorrect += 1
            else if result == 'fail' : isFail += 1
    return '인식률 ' + str(isCorrect / (count*5) * 100) + '% | 실패율 ' + str(isFail / (count*5) * 100) + '%'

# 인식률 계산
count = 3
compare(count)

