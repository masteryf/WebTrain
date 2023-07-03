import socket
import psutil
import GPUtil
from TrainCSV.train import Train_From_CSV
from Prediction.Prediction import prediction

# 服务器地址和端口
server_address = ('localhost', 8888)

# 创建TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)

print("等待客户端连接...")

while True:
    # 接受客户端连接
    client_socket, client_address = server_socket.accept()
    print("客户端已连接:", client_address)

    # 接收客户端发送的指令
    command = client_socket.recv(1024).decode()

    if command == '1':
        print("接收文件并训练")
        client_socket.sendall("训练".encode())
        # 接收文件
        file_data = client_socket.recv(1024)
        with open('Data/data.csv', 'wb') as file:
            while file_data:
                file.write(file_data)
                file_data = client_socket.recv(1024)

        print("接收完成")
        # 执行代码块1
        filename = Train_From_CSV(path='Data/data.csv')

        # 返回处理后的文件
        with open(filename, 'rb') as file:
            file_data = file.read()
            client_socket.sendall(file_data)

    elif command == '2':
        print("接收文件并预测")
        client_socket.sendall("预测".encode())
        # 接收文件
        file_data = client_socket.recv(1024)
        with open('Data/data.csv', 'wb') as file:
            while file_data:
                file.write(file_data)
                file_data = client_socket.recv(1024)

        # 执行代码块2
        output = prediction(path='Data/data.csv')
        client_socket.sendall(output.encode())

        # # 返回处理后的文件
        # with open('processed_file.txt', 'rb') as file:
        #     file_data = file.read()
        #     client_socket.sendall(file_data)

    elif command == '3':
        print("获取设备信息...")
        # 获取设备信息
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # 获取GPU信息
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                '名称': gpu.name,
                '显存总量': gpu.memoryTotal,
                '显存占用率': gpu.memoryUtil * 100,
                'GPU占用率': gpu.load * 100
            })

        # 构造设备信息的字符串
        device_info = "CPU占用率: {}%\n内存占用率: {}%\n\n".format(cpu_usage, memory_usage)
        device_info += "GPU信息:\n"
        for gpu in gpu_info:
            device_info += "名称: {}\n显存总量: {}\n显存占用率: {}%\nGPU占用率: {}%\n\n".format(
                gpu['名称'], gpu['显存总量'], gpu['显存占用率'], gpu['GPU占用率']
            )

        # 返回设备信息
        client_socket.sendall(device_info.encode())

    else:
        print("未知指令")

    # 关闭连接
    client_socket.close()

# 关闭服务器socket
server_socket.close()
