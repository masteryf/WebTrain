import torch
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1, 1)  # 调整目标张量的形状以匹配模型输出
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # 将一批的损失相加

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
