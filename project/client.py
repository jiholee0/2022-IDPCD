
# client
# 역할 : server로부터 페이지 번호와 전환 여부를 받아 올바른 제스처를 인식한 후 결과를 전송한다.

import socket, datetime, json

import sys
sys.path.append("/recog_gesture")
import recog_gesture as recog_gesture

HOST = '127.0.0.1' # local 호스트 사용
PORT = 10000 # 10000번 포트 사용
# 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 접속
client_socket.connect((HOST, PORT))

cliData = {
    'isComplete' : False,
    'curPageNum' : 1,
    'result' : ''
}
while cliData.get('isComplete') == False :
    # 제스처 인식 결과 가져오기
    result = recog_gesture.recog_gesture()

    # 데이터 서버에 전송
    cliData.update(result=result)
    sendData = json.dumps(cliData)
    client_socket.send(bytes(sendData, 'utf-8'))

    # 서버에게 데이터 받음
    data = client_socket.recv(1024)
    print(data)
    recvData = json.loads(data)
    print('recvData : ',recvData, datetime.datetime.now())
    cliData.update( isComplete=bool(recvData.get('isComplete')), curPageNum=int(recvData.get('curPageNum')))


client_socket.close()
