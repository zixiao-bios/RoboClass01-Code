import torch
import torch.nn as nn
import torch.nn.functional as F


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
