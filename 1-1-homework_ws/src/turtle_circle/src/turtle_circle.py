#! /usr/bin/env python
import rospy
import random
import math
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, SpawnRequest, SpawnResponse


if __name__ == '__main__':
    rospy.init_node("spawn_by_py")

    # 给服务传递的参数
    request = SpawnRequest()
    request.x = 8 * random.random()
    request.y = 8 * random.random()
    request.theta = math.pi * random.uniform(-1, 1)
    request.name = "own_turtle_1"

    # 请求服务
    rospy.loginfo('waitting for service: /spawn ...')
    rospy.wait_for_service('/spawn')

    try:
        service = rospy.ServiceProxy("/spawn", Spawn)
        response: SpawnResponse = service(request)
    except rospy.ServiceException as e:
        rospy.logerr(f'Service call failed: {e}')
        exit(-1)

    rospy.loginfo(f"spawn new turtle: {response.name}")

    # 创建速度控制 Topic 发布者
    pub = rospy.Publisher(request.name+'/cmd_vel', Twist, queue_size=10)

    # 发布的消息变量
    control_msg = Twist()
    control_msg.linear.x = 5 * random.random()
    control_msg.angular.z = random.choice([1, -1]) * random.uniform(0.5, 5)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(control_msg)
        pub.publish(control_msg)
        rate.sleep()
