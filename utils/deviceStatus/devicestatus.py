import psutil
import subprocess
import cpuinfo
import uuid
import socket

def getCPUUsage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def getMemoryUsage():
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    return memory_usage

def getMamorysize(type = 'left'):
    memory = psutil.virtual_memory()
    memory_total = memory.total // (1024 ** 3)  # 转换为GB
    memory_used = memory.used // (1024 ** 3)  # 转换为GB
    if type == 'left':
        return  memory_total-memory_used
    if type == 'used':
        return memory_used
    if type == 'total':
        return memory_total

def getGPUUsage():
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                        '--format=csv,nounits,noheader']).decode().strip().split('\n')
    gpu_usage = [float(info.split(',')[0]) for info in gpu_info]
    return gpu_usage

def getGPUMemoryUsed():
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                        '--format=csv,nounits,noheader']).decode().strip().split('\n')
    gpu_memory_used = [int(info.split(',')[1]) for info in gpu_info]
    return gpu_memory_used

def getGPUTemperature():
    gpu_temperature_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,nounits,noheader']).decode().strip().split('\n')
    gpu_temperature = [int(temp) for temp in gpu_temperature_info]
    return gpu_temperature

def getCPUTemperature():
    cpu_temp = cpuinfo.get_cpu_info()['hz_actual']
    return cpu_temp

def getDeviceid(type = 'hostname'):
    if type == 'hostname':
        return socket.gethostname()
    if type == 'mac':
        node = uuid.getnode()
        mac = uuid.UUID(int=node).hex[-12:]
        return ':'.join(mac[i:i + 2] for i in range(0, 12, 2))