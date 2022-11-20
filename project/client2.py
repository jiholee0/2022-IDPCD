import socket, cv2


HOST = '127.0.0.1' # local 호스트 사용
PORT = 10000 # 10000번 포트 사용
# 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 접속
client_socket.connect((HOST, PORT))


while True :
    client_socket.send('2'.encode('utf-8'))
    receive = client_socket.recv(100).decode()
    print('recieve : |' + receive+'|')
    
    sendData = input("input data :")
    client_socket.send(sendData.encode('utf-8'))
    receive = client_socket.recv(100).decode()
    print('recieve : |' + receive+'|')

client_socket.close()
