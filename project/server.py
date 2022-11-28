
# server
# 역할 : client로부터 제스처 인식 결과를 받아 키오스크 화면을 제어한 후 결과를 전송한다.

import socket, threading
from flask import render_template
import json, datetime, cv2

orders = []
pickUpYn = False
cashYn = False
def ctrlKiosk(recvData) -> json :
    global orders
    global pickUpYn
    global cashYn
    global openImage
    curPageNum = int(recvData.get('curPageNum'))
    nextPageNum = curPageNum + 1
    result = recvData.get('result')
    src = ""

    if curPageNum == 0 : # 초기 화면 띄우기
        if result == 'SERVER' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/001.png"
    elif curPageNum == 1 : # 초기 화면
        if result == 'OK' : # 다음 화면으로
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/002.png"
    elif curPageNum == 2 : # 주문 메뉴의 종류 선택
        if result == 'ZERO' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/003.png"
        # elif result == 'ONE' :
        #     src = ""
        # elif result == 'TWO' :
        #     src = ""
    elif curPageNum == 3 :
        if result == 'ZERO' :
            order = {
                'menu': '치즈버거',
                'count' : 1,
                'price' : 3900
            }
            orders.append(order)
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/004.png"
        # elif result == 'ONE' :
        #     src = ""
        # elif result == 'TWO' :
        #     src = ""
    elif curPageNum == 4 :
        if result == 'OK' :
            order = {
                'menu': '콜라, 감자튀김',
                'count' : 1,
                'price' : 2000
            }
            orders.append(order)
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/005.png"
        elif result == 'FIVE' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/005.png"
    elif curPageNum == 5 :
        if result == 'OK' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/002.png"
            nextPageNum = 2
        elif result == 'FIVE' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/006.png"
    elif curPageNum == 6 :
        if result == 'ONE' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/007.png"
        elif result == 'TWO' :
            pickUpYn = True
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/007.png"
    elif curPageNum == 7 :
        if result == 'ONE' :
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/008.png"
        elif result == 'TWO' :
            cashYn = True
            src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/008.png"
    elif curPageNum == 8 :
        if result == 'OK' :
            print('주문 내역 :',orders)
            recvData.update(isComplete=True)

    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if image is None :
        print('Image load failed...Please gesture one more time.')
        recvData.update(result='FAIL')
    else :
        cv2.destroyAllWindows
        openImage = cv2.resize(image, dsize=(300,420), interpolation=cv2.INTER_AREA)
        recvData.update(curPageNum=int(nextPageNum))
        cv2.imshow("kiosk", openImage)
        cv2.waitKey(1)
    return recvData

def binder(client_socket, addr):
    # 커넥션이 되면 접속 주소가 나온다.
    print('Connected by', addr)
    try:
        initData = {
            'isComplete' : False,
            'curPageNum' : 0,
            'result' : 'SERVER'
        }
        ctrlKiosk(initData)
        while True :

            # 클라이언트에게 데이터 받음
            data = client_socket.recv(1024)
            recvData = json.loads(data)
            print('recvData : ',recvData, datetime.datetime.now())

            # 키오스크 화면 제어
            servData = ctrlKiosk(recvData)

            # 클라이언트에게 데이터 전송
            sendData = json.dumps(servData)
            client_socket.send(bytes(sendData, 'utf-8'))

    except:
    # 접속 해제시 except
        print("except : " , addr)
    finally:
    # 종료
        client_socket.close()


# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 10000번 포트 사용
server_socket.bind(('127.0.0.1',10000))

server_socket.listen(1)

try:
    # 클라이언트가 접속하기 전까지 서버는 실행되야하기 때문에 무한 루프 사용
    while True:
        client_socket, addr = server_socket.accept()
    # 쓰레드 사용해서 대기
        th = threading.Thread(target=binder, args = (client_socket,addr))
        th.start()
except:
    print("server")
finally:
    # 종료
    server_socket.close()
