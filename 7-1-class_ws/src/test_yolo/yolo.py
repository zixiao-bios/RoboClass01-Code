from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np


current_dir_path = Path(__file__).resolve().parent

img_path = current_dir_path / 'imgs' / 'traffic.jpg'

yolo_weight_dir = '/root/yolo/weights/yolov8l.pt'


def main():
    # 实例化YOLO模型
    model = YOLO(yolo_weight_dir)

    # 读取本地图片
    input_img = Image.open(img_path).convert('RGB')

    # 将图片转为ndarray形式（矩阵）
    img_array = np.array(input_img)

    # 调用模型进行目标检测，测到结果（长度为1的数组，因此取索引为0的元素）
    result = model.predict(input_img, iou=0.999)[0]

    # 从结果中取出边界框
    boxes = result.boxes

    # 边界框的形状为(n, 4)
    n = boxes.shape[0]

    # 每种类别目标的计数
    count = {}

    for i in range(n):
        # 取第i个边界框，并转为int格式数据
        xyxy = boxes.xyxy[i].int()

        # 取第i个边界框的类别名称
        label = model.names[boxes.cls[i].item()]

        # 类别计数
        if label not in count:
            count[label] = 0
        count[label] += 1

        print(xyxy)
        print(label)
        print()

        # 用切片的方式取出边界框区域的图像
        obj_img = img_array[xyxy[1].item():xyxy[3].item(), xyxy[0].item():xyxy[2].item(), :]

        # 创建图像实例，并保存
        im = Image.fromarray(obj_img)
        im.save(f'{label}_{count[label]}.jpg')

    # 将检测结果渲染出来
    im_array = result.plot()  # plot a BGR numpy array of predictions

    # 创建图像实例，并保存
    im = Image.fromarray(im_array[:, :, ::-1])  # RGB PIL image
    im.save('result_iou0.999.jpg')  # save image


if __name__ == '__main__':
    main()
