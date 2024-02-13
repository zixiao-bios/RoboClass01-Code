import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from mlp import SimpleMLP

# 权重文件的文件名
weights_file = '1707816318_epoch_4.pt'

# 获取权重文件的路径
current_dir_path = Path(__file__).resolve().parent
model_weights_path = current_dir_path / 'weights' / weights_file

# 输入图片文件夹的路径
input_imgs_dir = current_dir_path / '..' / 'data' / 'my_write_num'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input_data(input_imgs_dir, img_size):
    # list，存储所有输入图片的路径
    imgs_path = input_imgs_dir.glob('*.jpg')
    
    # 转换输入图片的流水线
    transform = transforms.Compose([
        # 变为灰度图
        transforms.Grayscale(), 
        # 修改尺寸
        transforms.Resize(img_size, antialias=True), 
    ])

    # 读取输入图片，并进行预处理
    input_imgs = []
    for img_path in imgs_path:
        # 读取图片文件
        img = read_image(str(img_path))

        # 应用transform流水线到图片
        img = transform(img)

        # 从 0-255 归一化为 0-1
        img = img.type(torch.float32) / 255

        # 灰度反转
        img = img * -1 + 1
        
        input_imgs.append(img)

    # (n, 1, 28, 28)
    data = torch.stack(input_imgs).to(device)
    return data


def main():
    data = get_input_data(input_imgs_dir, (28, 28))

    writer = SummaryWriter(comment=f'_predict_{weights_file}')

    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    with torch.no_grad():
        # (b, 10)
        output = model(data)

        # (b, 1)
        pred = output.argmax(dim=1, keepdim=True)

    for i in range(10):
        mask = (pred.view(-1) == i)
        if mask.sum() > 0:
            writer.add_images(f'num={i}', data[mask])


if __name__ == '__main__':
    main()
