#! /usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Float64


def measure(data, params):
    std_dev = params[0]
    pub: rospy.Publisher = params[1]

    # 添加均值为0，指定标准差的高斯噪声，用于模拟传感器误差
    measure = data.data + np.random.normal(0, std_dev)
    pub.publish(measure)


def main():
    rospy.init_node('gps_sensor')

    # 获取该传感器的标准差
    std_dev = rospy.get_param('/gps_std_dev')

    pub = rospy.Publisher('p_gps', Float64, queue_size=10)

    # topic订阅者，并向回调函数传递两个参数
    rospy.Subscriber('p', Float64, measure, (std_dev, pub))
    rospy.spin()


if __name__=='__main__':
    main()
