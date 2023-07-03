# 运行Server.py来启动



## 训练方法：

导入：

```python
from TrainCSV.train import Train_From_CSV
```

使用：
```python
filename = Train_From_CSV(path='Data/data.csv')
```
其中，path参数为训练文件的地址，返回值为训练完成的模型权重的地址，默认在weights目录下

## 预测方法：

导入：

```python
from Prediction.Prediction import prediction
```

使用：
```python
output = prediction(path='Data/data.csv')
```
其中，path参数为预测参数文件的地址，返回值为故障种类