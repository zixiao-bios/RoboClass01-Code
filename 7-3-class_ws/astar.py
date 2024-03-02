from queue import PriorityQueue

class Node:
    def __init__(self, parent=None, position=None):
        # 当前位置的父节点，用于在找到终点后回溯路径
        self.parent = parent

        # 当前节点的位置
        self.position = position

        # 从起点到当前节点的成本
        self.g = 0
        # 从当前节点到终点的估计成本
        self.h = 0
        # 总成本
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self) -> int:
        return hash(self.position)


def astar(grid, start, end):
    # 起点和终点节点
    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))
    
    # 开放列表，用优先队列存储待扩展节点
    open_list = PriorityQueue()

    # 关闭列表，存储已经探索过的节点
    closed_set = set()
    
    # 将起点添加到开放列表
    open_list.put((start_node.f, start_node))
    
    # 循环直到找到路径或开放列表为空
    while not open_list.empty():
        # 获取当前节点
        current_node = open_list.get()[1]
        
        # 如果找到终点
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # 返回逆序路径
        
        # 将当前节点移动到关闭列表
        closed_set.add(current_node)
        
        # 生成子节点
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 相邻位置
            # 获取新节点位置
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            
            # 确保位置在范围内
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0]) - 1) or node_position[1] < 0:
                continue
            
            # 确保位置可行走
            if grid[node_position[0]][node_position[1]] != 0:
                continue
            
            # 创建新节点
            new_node = Node(current_node, node_position)
            
            # 如果新节点已在关闭列表中，跳过
            if new_node in closed_set:
                continue
            
            # 计算新节点的成本
            new_node.g = current_node.g + 1
            new_node.h = ((new_node.position[0] - end_node.position[0]) ** 2) + ((new_node.position[1] - end_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h
            
            # 检查新节点是否已在开放列表中，并且检查其成本是否更低
            if (new_node.f, new_node) in open_list.queue:
                continue
            
            # 将新节点添加到开放列表
            open_list.put((new_node.f, new_node))

    return None  # 如果没有找到路径，则返回None

if __name__=='__main__':
    # 定义栅格地图 (0 = 可走, 1 = 障碍物)
    grid = [[0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]]

    # 定义起点和终点
    start = [0, 0]
    end = [4, 4]

    # 获取路径
    path = astar(grid, start, end)
    print(path)

    for each in path:
        grid[each[0]][each[1]] = 2
    
    for line in grid:
        print(line)
