import torch
import torch.nn.functional as F
import time
from utils.webConnect.webconnect import *
def train(model, device, train_loader, optimizer, epoch, sock):
    model.train()
    start_time = time.time() # 记录开始时间
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float() # 将目标张量转换为 Float 类型
        target = target.view(-1, 1) # 调整目标张量的形状以匹配模型输出
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target) # 修改损失函数为均方误差
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('TrainCSV Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            send_msg('TrainCSV Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()),sock)
    end_time = time.time() # 记录结束时间
    # print("Training time for epoch {}: {:.2f} seconds".format(epoch, end_time - start_time)) # 计算并打印用时


