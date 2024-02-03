#! /usr/bin/env python  
import rospy
import numpy as np
from std_msgs.msg import Float64


# z_k
gps_value = None
odom_value = None

# u_k
a_value = None


class KalmanFilter():
    def __init__(self, x_dim, u_dim, z_dim, A, B, H, Q, R, x0, P0) -> None:
        # 数据检验
        assert A.shape[0] == x_dim and A.shape[1] == x_dim, "A.shape must be x_dim * x_dim!"
        assert B.shape[0] == x_dim and B.shape[1] == u_dim, "B.shape must be x_dim * u_dim!"
        assert H.shape[0] == z_dim and H.shape[1] == x_dim, "H.shape must be z_dim * x_dim!"
        assert Q.shape[0] == x_dim and Q.shape[1] == x_dim, "Q.shape must be x_dim * x_dim!"
        assert R.shape[0] == z_dim and R.shape[1] == z_dim, "R.shape must be z_dim * z_dim!"
        assert x0.shape[0] == x_dim, "x0.shape[0] must be x_dim!"
        assert P0.shape[0] == x_dim and P0.shape[1] == x_dim, "P0.shape must be x_dim * x_dim!"

        # 赋值
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        # 状态的后验估计
        self.x = x0

        # 状态的先验估计
        self.x_ = None

        # 后验估计的协方差
        self.P = P0

        # 先验估计的协方差
        self.P_ = None

        # 卡尔曼增益
        self.K = None
    
    def predict(self, u=None):
        # 预测，传入控制量 u_k

        # 先验估计
        self.x_ = self.A @ self.x
        if u is not None:
            assert u.shape[0] == self.u_dim, "u.shape[0] must be u_dim!"
            self.x_ += self.B @ u
        
        # 先验估计的协方差
        self.P_ = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        #更新，传入测量值 z_k

        assert z.shape[0] == self.z_dim, "z.shape[0] must be z_dim!"

        # 计算卡尔曼增益
        self.K = (self.P_ @ self.H.T) @ np.linalg.inv(self.H @ self.P_ @ self.H.T + self.R)

        # 后验估计
        self.x = self.x_ + self.K @ (z - self.H @ self.x_)

        # 后验估计的协方差
        self.P = (np.identity(self.x_dim) - self.K @ self.H) @ self.P_
        
        # 返回后验估计，滤波器的输出
        return self.x


def get_gps(data):
    global gps_value
    gps_value = data.data


def get_odom(data):
    global odom_value
    odom_value = data.data


def get_a(data):
    global a_value
    a_value = data.data


def main():
    rospy.init_node('filter')
    dt = rospy.get_param('/dt')

    x_dim = 2
    u_dim = 1
    z_dim = 2

    A = np.array([[1, dt], 
                  [0, 1]])
    B = np.array([[dt * dt / 2], 
                  [dt]])
    H = np.identity(2)

    # 定义 Q 矩阵
    p_std_dev = rospy.get_param('~p_std_dev')
    v_std_dev = rospy.get_param('~v_std_dev')
    Q = np.array([[p_std_dev ** 2, 0], 
                 [0, v_std_dev ** 2]])
    
    # 定义 R 矩阵
    gps_std_dev = rospy.get_param('/gps_std_dev')
    odom_std_dev = rospy.get_param('/odom_std_dev')
    R = np.array([[gps_std_dev ** 2, 0], 
                  [0, odom_std_dev ** 2]])
    
    # 定义初值
    x0 = np.array([[0], 
                   [0]])
    P0 = np.array([[1, 0], 
                   [0, 1]])
    
    kf = KalmanFilter(x_dim, u_dim, z_dim, 
                      A, B, H, Q, R, 
                      x0, P0)

    # 订阅 z_k 和 u_k
    rospy.Subscriber('p_gps', Float64, get_gps)
    rospy.Subscriber('v_odom', Float64, get_odom)
    rospy.Subscriber('a', Float64, get_a)

    # 滤波器估计值的发布者
    p_pub = rospy.Publisher('p_est', Float64, queue_size=10)
    v_pub = rospy.Publisher('v_est', Float64, queue_size=10)

    # 轮询频率要设置高一些
    rate = rospy.Rate(5 / dt)
    i = 0

    global gps_value, odom_value, a_value
    while not rospy.is_shutdown():
        # 只有新一轮的 u_k 和 z_k 都收到后，才进行滤波
        if gps_value is not None and odom_value is not None and a_value is not None:
            rospy.loginfo(f'{i}, {gps_value}, {odom_value}, {a_value}')
            i += 1

            # 预测
            kf.predict(np.array([[a_value]]))

            # 更新，得到估计值
            x = kf.update(np.array([[gps_value], 
                                    [odom_value]]))
            
            # 发布估计值 Topic 数据
            p_pub.publish(x[0, 0])
            v_pub.publish(x[1, 0])

            # 等待下一轮数据接收
            gps_value = None
            odom_value = None
            a_value = None
        
        rate.sleep()


if __name__ == '__main__':
    main()
