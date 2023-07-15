# 运行Server.py来启动

## 性能：
| 设备                      | 单论epoch时间 |
|-------------------------|-----------|
| ryzen9_5900HX    | 110s      | 
| NVIDIA Geforce RTX 4090 | 10.6s     | 
| NVIDIA Tesla A40        | 13.6s     |
| NVIDIA A100 80GB        | 8.2s      | 
5-10轮在训练集上即可达到大部分精度![acc.png](photos%2Facc.png)
## 训练方法：

导入：

```python
from TrainCSV.train import Train_From_CSV
from utils.csvnorm.csvnorm import process_csv
```
| 参数       | 值            | 描述                                       |
|----------|--------------|------------------------------------------|
| in_path  | string       | 输入的训练文件地址                                |
| out_path | string       | 输出的模型文件地址                                |
| sock     | socket       | 用于返回训练数据的socket                          |
| echo     | bool         | 是否返回训练数据                                 |
| BATCH_SIZE  | int          | 批次大小，默认为32                               |
| EPOCHS | int          | 训练轮数，默认为20                               |
| DEVICE     | torch.device | 会自动选择cpu或者cuda，可不填，除非需要使用directml或者国产npu |
| LEARNRATE     | int          | 学习率，默认为1e-5                              |
使用：
```python
process_csv(input_file=data_path+file_name,output_file=data_path+"norm_"+file_name)#归一化数据
filename = Train_From_CSV(in_path=data_path+"norm_"+file_name, sock=sock, echo=True, out_path=weight_path)
```
返回值为训练好的模型地址

## 预测方法：

导入：

```python
from Prediction.Prediction import prediction
```
| 参数    | 值            | 描述        |
|-------|--------------|-----------|
| path  | string       | 输入的预测文件地址 |
| weight_path | string       | 模型文件地址 |
使用：
```python
label = prediction(path=data_path+file_name,weight_path = "")
```
返回值为标签

## 网络传输方法：
导入：

```python
from utils.webConnect.webconnect import *
```
使用：
```python
get_response(sock)#接收字符串
```
```python
send_msg(data, sock)#发送字符串
```
```python
get_file(fileName, path, sock)#接收文件
```
```python
send_file(fileName, path, sock)#发送文件
```

## 获取设备状态：
导入：

```python
from utils.deviceStatus.devicestatus import *
```
使用：
```python
getCPUUsage()#获取CPU占用
```
```python
getMemoryUsage()#获取内存占用
```
```python
getMamorysize(type = 'left')#获取内存容量
```
| 参数    | 描述   |
|-------|------|
 | left  | 剩余内存 |
| used  | 已用内存 |
| total | 总内存  |

```python
getGPUUsage()#获取GPU占用
```
```python
getGPUMemoryUsed()#获取显存占用
```
```python
getGPUTemperature()#获取GPU温度
```
```python
getCPUTemperature()#获取CPU温度
```
```python
getDeviceid(type = 'hostname')#获取设备名
```
| 参数       | 描述    |
|----------|-------|
 | hostname | 主机名   |
| mac      | mac地址 |