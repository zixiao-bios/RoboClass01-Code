import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


# 定义一个简单的多层感知机模型
# pytorch 中所有自定义的神经网络类，都要继承 nn.Module
class SimpleMLP(nn.Module):
    def __init__(self):
        # 调用基类的构造函数
        super(SimpleMLP, self).__init__()

        # 第一层全连接层，输入784（28x28图像），输出256
        self.fc1 = nn.Linear(784, 256)

        # 第二层全连接层，输入256，输出128
        self.fc2 = nn.Linear(256, 128)

        # 第三层全连接层，输入128，输出10（类别数）
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 将图像展平为一维向量
        x = torch.flatten(x, 1)

        # 第一层全连接后使用ReLU激活函数
        x = F.relu(self.fc1(x))

        # 第二层全连接后使用ReLU激活函数
        x = F.relu(self.fc2(x))

        # 第三层全连接输出
        x = self.fc3(x)

        # 在使用nn.CrossEntropyLoss时，不需要在这里应用Softmax
        return x


# 训练模型
def train(model, device, train_loader, optimizer, epoch, loss_func):
    # 设置网络为训练模式
    model.train()

    # enumerate用法：https://www.runoob.com/python/python-func-enumerate.html
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据传送到指定设备上（如GPU）
        data, target = data.to(device), target.to(device)

        # 将网络中的梯度清零
        optimizer.zero_grad()

        # 进行一次推理
        output = model(data)

        # 调用损失函数，计算损失值
        loss = loss_func(output, target)

        # 反向传播，计算loss的梯度
        loss.backward()

        # 使用网络中的梯度更新参数
        optimizer.step()

        # 每100次循环打印一次
        if batch_idx % 100 == 0:
            print(f"训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")

# 测试模型
def test(model, device, test_loader, loss_func):
    # 设置网络为评估模式
    model.eval()

    # 测试集上的损失
    test_loss = 0
    correct = 0

    # 关闭自动求导，测试模式下不更新参数，不需要求导
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()

            # 用模型的输出，计算每个输入的预测类别
            # `argmax(dim=1)`找到每个样本最大概率的索引，即模型预测的类别
            # `keepdim=True`保持输出的维度，便于后续的比较操作
            pred = output.argmax(dim=1, keepdim=True)

            # 比较预测结果和真实标签
            # `eq`函数比较预测的类别（pred）和真实的类别（target）是否相等，并返回一个布尔值数组
            # `view_as(pred)`确保target的形状与pred相同
            # `sum().item()`计算相等的元素数量，即正确预测的数量，并将其转换为Python的数字
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"\n测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


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
    model = SimpleMLP().to(device)

    # 定义损失函数，这里使用交叉熵
    loss_func = nn.CrossEntropyLoss()

    # 优化器，负责优化网络参数，这里使用SGD算法
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 运行训练和测试
    for epoch in range(1, 6):  # 总共训练5轮
        train(model, device, train_loader, optimizer, epoch, loss_func)
        test(model, device, test_loader, loss_func)


if __name__ == '__main__':
    main()
