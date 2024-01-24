#! /usr/bin/env python
import rospy
from learn_service.srv import AddTwoNum, AddTwoNumRequest, AddTwoNumResponse


# AddTwoNum 的处理函数，参数为 AddTwoNumRequest，返回 AddTwoNumResponse
def handle_add_two_num(req: AddTwoNumRequest):
    sum = req.A + req.B
    rospy.loginfo(f'{req.A} + {req.B} = {sum}')
    return AddTwoNumResponse(sum)


if __name__ == '__main__':
    rospy.init_node('add_two_num_server')

    # 创建 Service 服务器，类型为 AddTwoNum，处理函数为 handle_add_two_num
    rospy.Service('add_two_num', AddTwoNum, handle_add_two_num)
    rospy.loginfo('add_two_num server ready.')
    rospy.spin()
