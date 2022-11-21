
# server
# 역할 : client로부터 제스처 인식 결과를 받아 키오스크 화면을 제어한 후 결과를 전송한다.

import socket, threading
from flask import render_template
import json, datetime, cv2


def ctrlKiosk(resvData) -> json :
    src = "C:/Users/NICE-DNB/Desktop/2022-IDPCD/project/kiosk/image/00"+resvData.get('result')+".png"

    cv2.destroyAllWindows
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if image is None :
        print('Image load failed')
    else :
        openImage = cv2.resize(image, dsize=(300,420), interpolation=cv2.INTER_AREA)
        cv2.imshow("kiosk", openImage)
        cv2.waitKey(1)
        resvData.update(curPageNum=int(resvData.get('result')), isComplete=True)
    return resvData

def binder(client_socket, addr):
    global servData
    # 커넥션이 되면 접속 주소가 나온다.
    print('Connected by', addr)
    try:
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
            print('sendData : ',sendData, datetime.datetime.now())

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
