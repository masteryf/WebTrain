import json
def get_response(sock):
    response_length_bytes = sock.recv(4)
    response_length = int.from_bytes(response_length_bytes, 'big')
    response_data = sock.recv(response_length)
    return response_data
def send_msg(data, sock):
    data_str = json.dumps(data)
    # 发送数据长度
    length = len(data_str)
    length_bytes = length.to_bytes(4, 'big')
    sock.sendall(length_bytes)
    # 发送数据
    sock.sendall(data_str.encode('utf-8'))
def get_file(fileName, path, sock):
    with open(path + fileName, "wb") as file:
        while True:
            # 接收服务器的响应
            response_length_bytes = sock.recv(4)
            response_length = int.from_bytes(response_length_bytes, 'big')
            response_data = sock.recv(response_length)
            if response_data.decode('utf-8', errors='ignore') == "close":
                file.close()
                break
            file.write(response_data)
def send_file(fileName, path, sock):
    return