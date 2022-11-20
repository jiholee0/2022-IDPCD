import socket, threading
from flask import render_template
import time

# 데이터
isComplete = False
curPageNum = -1
result = ''
 
def recvValue(clientNum, result) :
    if clientNum == '1' :
        if result == 'ZERO' : 
            client_socket.sendall(curPageNum.encode('utf-8'));
            client_socket.sendall(result.get(result).encode('utf-8'));
            return 'SUCCESS'
    else : 
        return result

def binder(client_socket, addr):
    # 커넥션이 되면 접속 주소가 나온다.
    print('Connected by', addr);
    try:
        while True:
            # 1024byte 이하의 데이터  수신
            data = client_socket.recv(1024);
            clientNum = data.decode();
            print('Received from', addr, clientNum);    
            client_socket.sendall(clientNum.encode('utf-8'));
            while result.length > 0 :
                print('client1')
                if clientNum == '1' :
                    # result를 client 1에게 전송한다.
                        client_socket.sendall(result.encode('utf-8'));
                        # 전환 후 페이지 번호를 client 1에게서 받는다.
                        data = client_socket.recv(1024);
                        curPageNum = data.decode();
                        # 화면 전환 여부를 client 1에게서 받는다.
                        data = client_socket.recv(1024);
                        isComplete = data.decode();
                        # 수신된 메시지를 콘솔에 출력한다.
                        print('Received from', addr, curPageNum, isComplete);

            data = client_socket.recv(1024);
            result = data.decode();
            # 수신된 메시지를 콘솔에 출력한다.
            print('Received from', addr, result);    
            client_socket.sendall(result.encode('utf-8'));

    except:
    # 접속 해제시 except
        print("except : " , addr);
    finally:
    # 종료
        client_socket.close();


# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);

# 10000번 포트 사용
server_socket.bind(('127.0.0.1',10000));

server_socket.listen(1);

try:
    # 클라이언트가 접속하기 전까지 서버는 실행되야하기 때문에 무한 루프 사용
    while True:
        client_socket, addr = server_socket.accept();
    # 쓰레드 사용해서 대기
        th = threading.Thread(target=binder, args = (client_socket,addr));
        th.start();
except:
    print("server");
finally:
    # 종료
    server_socket.close();
