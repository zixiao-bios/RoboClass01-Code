#! /usr/bin/env python
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv

if __name__ == '__main__':
    rospy.init_node('turtle_tf_listener')

    # 调用 Service 生成海龟
    rospy.wait_for_service('spawn')
    spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    spawner(4, 2, 0, 'turtle2')

    # 定义发布海龟速度的 Topic
    turtle_vel = rospy.Publisher('turtle2/cmd_vel', geometry_msgs.msg.Twist, queue_size=1)

    # 定义 tf 监听器
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # 等待指定的 tf 变换可用
            listener.waitForTransform('/turtle2', '/turtle1', rospy.Time(0), rospy.Duration(1))

            # 查询 tf 树中，/turtle1 到 /turtle2 坐标系的变换（/turtle1 在 /turtle2 坐标系中的位姿）
            # 第三个参数为时间，即要查询哪个时刻的变换（tf tree可以记录和查询历史数据），rospy.Time(0)表示查询最新数据
            # 函数返回值为：([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])，分别是平移向量和四元数
            (trans, rot) = listener.lookupTransform('/turtle2', '/turtle1', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f'lookupTransform error: {e}')
            continue

        # 用相对位置计算角速度、线速度（也可以用别的计算方法，合理即可）
        angular = 4 * math.atan2(trans[1], trans[0])
        linear = 0.5 * math.sqrt(trans[0] ** 2 + trans[1] ** 2)

        # 发布速度控制指令 Topic
        cmd = geometry_msgs.msg.Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        turtle_vel.publish(cmd)

        rate.sleep()
