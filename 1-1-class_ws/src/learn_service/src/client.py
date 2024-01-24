#! /usr/bin/env python
import rospy
from learn_service.srv import AddTwoNum, AddTwoNumResponse, AddTwoNumRequest


def add_two_num_client(a, b):
    # 等待该 Service 在 Master 中注册
    rospy.loginfo('waitting for service: add_two_num ...')
    rospy.wait_for_service('add_two_num')

    # 捕获可能发生的错误
    try:
        # 请求服务
        service = rospy.ServiceProxy('add_two_num', AddTwoNum)

        request = AddTwoNumRequest()
        request.A = a
        request.B = b

        res: AddTwoNumResponse = service(request)
        return res.Sum
    except rospy.ServiceException as e:
        rospy.logerr(f'Service call failed: {e}')


if __name__ == '__main__':
    rospy.init_node('add_two_num_client')

    a = 2
    b = 3
    rospy.loginfo(f'request service: {a}+{b}')

    res = add_two_num_client(a, b)
    rospy.loginfo(f'get response: {res}')
