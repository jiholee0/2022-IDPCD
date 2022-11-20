
# client 1
# 역할 : result를 보고 키오스크 화면을 제어한다.
# server로부터 받는 value : result
# server에게 보내는 value : curPageNum, isComplete

curPageNum = -1
isComplete = True

import socket, cv2

HOST = '127.0.0.1' # local 호스트 사용
PORT = 10000 # 10000번 포트 사용
# 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 접속
client_socket.connect((HOST, PORT))

def openKiosk(result) :
    if result == 'ZERO' :
        pageNum = '001'
    src = "C:/Users/LJH/Desktop/2022-IDPCD/project/kiosk/image/"+pageNum+".png"
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if image is None :
        print('Image load failed')
    else :
        curPageNum = curPageNum + 1
        isComplete = True
        cv2.imshow("kiosk", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows

while True :
    isComplete = False
    client_socket.send('1'.encode('utf-8')) # client 1임을 나타내는 value 전송
    result = client_socket.recv(1024).decode() # server로부터 제스처 인식 결과값 receive
    print('receive : '+ result)
    openKiosk(result) # 키오스크 화면 전환 & 전송 값 설정
    print(curPageNum + ' ' + isComplete)

    client_socket.send(curPageNum.encode('utf-8')) # server에 값 전송
    client_socket.send(isComplete.encode('utf-8')) # server에 값 전송
    break
client_socket.close()
