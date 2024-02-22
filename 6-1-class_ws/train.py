import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time

from cnn import SimpleCNN, LeNet5


# 训练模型
def train(model, device, train_loader, optimizer, epoch, loss_func, writer: SummaryWriter = None):
    # 设置网络为训练模式
    model.train()

    # 本轮训练的平均loss
    train_loss = 0

    # enumerate用法：https://www.runoob.com/python/python-func-enumerate.html
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据传送到指定设备上（如GPU）
        # data.shape = (b, 1, 28, 28)
        # target.shape = (b, 10)
        data, target = data.to(device), target.to(device)

        # 将网络中的梯度清零
        optimizer.zero_grad()

        # 进行一次推理
        # output.shape = (b, 10)
        output = model(data)

        # 调用损失函数，计算损失值
        loss = loss_func(output, target)
        train_loss += loss.item()

        # 反向传播，计算loss的梯度
        loss.backward()

        # 使用网络中的梯度更新参数
        optimizer.step()

        # 写入tensorboard
        writer.add_scalar(f'loss_epoch_{epoch}', loss.item(), batch_idx)

        # 每100次循环打印一次
        if batch_idx % 100 == 0:
            print(f"训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
        
    return train_loss / len(train_loader)

# 测试模型
def test(model, device, test_loader, epoch, loss_func, writer: SummaryWriter = None):
    # 设置网络为评估模式
    model.eval()

    # 测试集上的损失
    test_loss = 0
    correct = 0

    # 关闭自动求导，测试模式下不更新参数，不需要求导
    with torch.no_grad():
        for data, target in test_loader:
            # data.shape = (b, 1, 28, 28)
            data, target = data.to(device), target.to(device)

            # (b, 10)
            output = model(data)

            test_loss += loss_func(output, target).item()

            # 用模型的输出，计算每个输入的预测类别
            # `argmax(dim=1)`找到每个样本最大概率的索引，即模型预测的类别
            # `keepdim=True`保持输出的维度，便于后续的比较操作
            # pred.shape = (b, 1)
            pred = output.argmax(dim=1, keepdim=True)

            # 比较预测结果和真实标签
            # `eq`函数比较预测的类别（pred）和真实的类别（target）是否相等，并返回一个布尔值数组
            # `view_as(pred)`确保target的形状与pred相同
            # `sum().item()`计算相等的元素数量，即正确预测的数量，并将其转换为Python的数字
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    for i in range(10):
        # pred.view(-1) 将 pred 调整为列向量, 即(b, 1) -> (b)
        # mask中为布尔值，表示对应位置图像的预测值是否为当前类别
        mask = (pred.view(-1) == i)

        # 如果当前类别下有图像，则用writer写入
        if mask.sum() > 0:
            # 用mask作为索引，即选取mask为true的位置对应的data
            writer.add_images(f'epoch_{epoch}, num={i}', data[mask])

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f"\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.0f}%)\n")
    return {'loss': test_loss, 'accuracy': accuracy}


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 载入MNIST数据集（从本地加载）
    # 获取该python文件的路径
    current_dir_path = Path(__file__).resolve().parent

    # 加载训练集，参数：数据集的本地路径、使用训练集还是测试集、不下载数据集、数据预处理流程
    train_dataset = datasets.MNIST(root=str(current_dir_path / '..' / 'data'), 
                                   train=True, 
                                   download=False, 
                                   transform=transforms.ToTensor())
    
    # 加载测试集，参数含义同上
    test_dataset = datasets.MNIST(root=str(current_dir_path / '..' / 'data'), 
                                  train=False, 
                                  download=False, 
                                  transform=transforms.ToTensor())

    # 从数据集创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 实例化模型并移至设备
    # model = SimpleCNN().to(device)
    model = LeNet5(10).to(device)

    # 定义损失函数，这里使用交叉熵
    loss_func = nn.CrossEntropyLoss()

    # 优化器，负责优化网络参数，这里使用SGD算法
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # tensorboard
    writer = SummaryWriter(comment='_train_LeNet5')

    # 记录模型结构图
    writer.add_graph(model, input_to_model=torch.rand(1, 1, 28, 28).to(device))

    weights_save_dir = current_dir_path / 'weights'
    weights_save_dir.mkdir(exist_ok=True)

    # 运行训练和测试
    for epoch in range(1, 6):  # 总共训练5轮
        # 进行一轮训练
        train_loss = train(model, device, train_loader, optimizer, epoch, loss_func, writer)

        # 写入本轮训练的loss
        writer.add_scalar('train_loss', train_loss, epoch)
        
        # 进行一轮测试
        test_data = test(model, device, test_loader, epoch, loss_func, writer)

        # 写入本轮测试的loss
        writer.add_scalar('test_loss', test_data['loss'], epoch)
        writer.add_scalar('accuracy', test_data['accuracy'], epoch)

        # 保存该epoch的模型权重
        torch.save(model.state_dict(), weights_save_dir / f'{int(time.time())}_epoch_{epoch}.pt')

    writer.close()


if __name__ == '__main__':
    main()
