import sys
sys.path.append("/recog_gesture")
import recog_gesture as recog_gesture
import time

kiosk_gesture = ['ZERO', 'ONE', 'TWO', 'FIVE', 'OK', 'GOOD']

def compare(gesture, count) -> str :
    isCorrect = 0
    isIncorrect = 0
    isFail = 0
    for i in range(0,count) :
        result = recog_gesture.recog_gesture()
        print(i+1, '. return value : ', result , time.time())
        if result == gesture : isCorrect += 1
        elif result == 'fail' : isFail += 1
        else : isIncorrect += 1
    return '인식률 ' + str(isCorrect / (count) * 100) + '% | 불일치율 ' + str(isIncorrect / count * 100) + '% | 실패율 ' + str(isFail / (count) * 100) + '%'

# 인식률 계산
count = 5
for gesture in kiosk_gesture :
    print(gesture, ':', compare(gesture, count))
    break

