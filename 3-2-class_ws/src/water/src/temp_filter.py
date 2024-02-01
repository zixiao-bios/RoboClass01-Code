#! /usr/bin/env python  
import rospy
from std_msgs.msg import Float64

e_mea = 0
e_est = 1
x = 0


def get_measure(data, pub: rospy.Publisher):
    global e_mea, e_est, x
    z = data.data
    k = e_est / (e_est + e_mea)
    x += k * (z - x)
    e_est *= (1 - k)

    pub.publish(x)
    rospy.loginfo(f'filtered temp: {x}')


def main():
    rospy.init_node('temp_filter')

    global e_mea, x
    e_mea = rospy.get_param('/std_dev')  # 测量误差
    x = rospy.get_param('~x0')  # 估计值的初始值

    pub = rospy.Publisher('temp_filtered', Float64, queue_size=10)

    rospy.Subscriber('temp_sensor', Float64, get_measure, pub)
    rospy.spin()


if __name__=='__main__':
    main()
