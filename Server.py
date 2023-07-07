import socket
from utils.webConnect.webconnect import *


# 服务器地址和端口号
server_address = ('localhost', 8000)

# 创建一个TCP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

receive = {
    "code": 10000,
}

info_msg = {
    "identifier": "",
    "filename": "",
    "totalChunks": 3,
    "chunkNumber": 4,
    "totalSize": 5,
    "file": "文件"
}

file_msg = {
    "code": 20001,
    "bytes": bytearray
}

path = ""
# 连接到服务器
sock.connect(server_address)

# 声明自己是数据接收端
send_msg(receive, sock)
res = json.loads(get_response(sock).decode())

if res['message'] == "200":
    print("与服务器建立连接成功！")
    while True:
        res = json.loads(get_response(sock).decode())
        print(res)

        if res['message'] == "train":
            # 接收文件
            file_name = res['fileName']
            get_file(file_name)
            print(file_name + "：文件已经接收")
            ## 训练文件

            ## post发送请求上传文件（分片）

        if res['message'] == "forecast":
            # 接收文件
            file_name = res['fileName']
            get_file(file_name, path, sock)
            print(file_name + "：文件已经接收")
            ## 训练文件

            ## 上传文件

        if res['message'] == "info":
            info_msg['gpu_info'].append({
                '名称': "cpu1",
                '显存总量': 1024,
                '显存占用率': 1024 * 100,
                'GPU占用率': 1024 * 100
            })
            send_msg(info_msg, sock)
            info_msg = []

        ## 监听消息
        res = get_response(sock)


else:
    print("与服务器建立连接失败")
