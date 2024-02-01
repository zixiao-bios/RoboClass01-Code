#! /usr/bin/env python  
import rospy
from std_msgs.msg import Float64


class Predicter:
    def __init__(self, r, boiling_temp, delta_t, initial_temp):
        self.r = r  # 加热率
        self.boiling_temp = boiling_temp  # 水的沸点
        self.delta_t = delta_t  # 测量间的时间间隔
        self.result = initial_temp  # 初始温度估计

    def predict(self):
        # 根据加热率和与沸点的差异预测下一步的温度
        self.result += self.delta_t * self.r * (self.boiling_temp - self.result)
        return self.result


class KalmanFilter:
    def __init__(self, r, boiling_temp, delta_t, initial_temp, initial_est_error, measurement_error):
        self.r = r  # 加热率
        self.boiling_temp = boiling_temp  # 水的沸点
        self.delta_t = delta_t  # 测量间的时间间隔
        self.estimate = initial_temp  # 初始温度估计
        self.estimated_error = initial_est_error  # 初始估计误差
        self.measurement_error = measurement_error  # 测量误差（传感器噪声的标准差）

    def predict(self):
        # 根据加热率和与沸点的差异预测下一步的温度
        self.estimate += self.delta_t * self.r * (self.boiling_temp - self.estimate)

    def update(self, measurement):
        # 计算卡尔曼增益
        K = self.estimated_error / (self.estimated_error + self.measurement_error)
        
        # 使用测量值更新估计值
        self.estimate += K * (measurement - self.estimate)
        
        # 更新估计误差
        self.estimated_error = (1 - K) * self.estimated_error

        return self.estimate


def get_temp_data(data, params):
    filtered_pub: rospy.Publisher = params[0]
    predict_pub: rospy.Publisher = params[1]
    kf: KalmanFilter = params[2]
    predicter: Predicter = params[3]

    # 单纯用模型预测
    x_predict = predicter.predict()

    # 卡尔曼滤波估计
    kf.predict()
    x = kf.update(data.data)

    predict_pub.publish(x_predict)
    filtered_pub.publish(x)
    rospy.loginfo(f'filtered temp: {x}')
    return x


def main():
    global err_mea
    rospy.init_node('temp_filter')

    # 设置模型参数
    r = 0.4  # 估计的加热率
    boiling_temp = 100  # 水的沸点
    delta_t = rospy.get_param('/delta_t')  # 测量之间的时间间隔
    initial_temp = 0  # 初始温度

    # 创建模型预测器实例
    predicter = Predicter(r, boiling_temp, delta_t, initial_temp)

    # 创建卡尔曼滤波器实例
    initial_est_error = 2  # 初始估计误差
    measurement_error = rospy.get_param('/std_dev')  # 测量误差
    kf = KalmanFilter(r, boiling_temp, delta_t, initial_temp, initial_est_error, measurement_error)

    filtered_pub = rospy.Publisher('temp_filtered', Float64, queue_size=10)
    predict_pub = rospy.Publisher('temp_predict', Float64, queue_size=10)

    rospy.Subscriber('water_temp', Float64, get_temp_data, (filtered_pub, predict_pub, kf, predicter))
    rospy.spin()


if __name__=='__main__':
    main()
