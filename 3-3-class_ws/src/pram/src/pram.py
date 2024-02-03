#! /usr/bin/env python  
import rospy
import numpy as np
from std_msgs.msg import Float64


class Car:
    def __init__(self):
        # 真实的初始位置
        self.p = 0
        self.v = 0
    
    def step(self, a, dt, p_std_dev=0, v_std_dev=0):
        # 理想情况下的运动模型
        self.p += self.v * dt + 0.5 * a * dt ** 2
        self.v += a * dt

        # 加入过程噪声
        self.p += np.random.normal(0, p_std_dev)
        self.v += np.random.normal(0, v_std_dev)


def main():
    rospy.init_node('pram')
    dt = rospy.get_param('/dt')

    # 加载加速度控制数组
    a_list = np.load(rospy.get_param('~accelerations_file'))

    # 真实的过程噪声
    p_std_dev = rospy.get_param('~p_std_dev')
    v_std_dev = rospy.get_param('~v_std_dev')

    # u_k, z_k 的发布者
    a_pub = rospy.Publisher('a', Float64, queue_size=10)
    p_pub = rospy.Publisher('p', Float64, queue_size=10)
    v_pub = rospy.Publisher('v', Float64, queue_size=10)

    pram = Car()
    i = 0
    rospy.sleep(2)
    rate = rospy.Rate(1 / dt)

    # 完成所有控制，或者节点关闭时退出循环
    while (i < len(a_list)) and (not rospy.is_shutdown()):
        # 对小车进行一次控制
        pram.step(a_list[i], dt, p_std_dev, v_std_dev)

        # 发布 u_k, z_k
        a_pub.publish(a_list[i])
        p_pub.publish(pram.p)
        v_pub.publish(pram.v)

        i += 1
        rate.sleep()


if __name__ == '__main__':
    main()
