# Project 1

In this project, we implement DFS, BFS and UCS to the graph so that all algorithms must fit the property of graphs.

```py 
def genericSearch(problem: SearchProblem, data_structure, is_priority_queue=False):
    """
    通用搜索函数，可用于深度优先搜索、广度优先搜索和一致代价搜索。
    :param problem: 搜索问题实例
    :param data_structure: 使用的数据结构（栈、队列或优先队列）
    :param is_priority_queue: 是否为优先队列
    :return: 到达目标状态的动作列表
    """
    visited = set() # the property of graphs
    currentState = problem.getStartState()
    if is_priority_queue:
        data_structure.update((currentState, []), 0)
    else:
        data_structure.push((currentState, [])) # [] stores the path

    while True:
        if data_structure.isEmpty():
            return []

        node, pathToNode = data_structure.pop()

        if problem.isGoalState(node):
            return pathToNode

        if node not in visited:
            visited.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                new_path = pathToNode + [action]
                if is_priority_queue:
                    cost = problem.getCostOfActions(new_path)
                    data_structure.update((successor, new_path), cost)
                else:
                    data_structure.push((successor, new_path))
                    
# The main difference between search methods lies in the stored data structure.
def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.
    """
    stack = util.Stack()
    return genericSearch(problem, stack)

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    return genericSearch(problem, queue)

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of the least total cost first."""
    prior_queue = util.PriorityQueue()
    return genericSearch(problem, prior_queue, is_priority_queue=True)
```

For A* search, if A* is not the consistency and admissibility, we have to expend nodes according to sum up heuristics and real cost with the most fewer values. We could expend nodes again.

```py
def aStarWithoutConsistency(problem, heuristic, prior_queue) -> List[Directions]:
    """It's the case for some graphs not fitting consistency especially q4. """
    while 1:
        if prior_queue.isEmpty():
            return []
        node, pathToNode = prior_queue.pop()
        if problem.isGoalState(node):
            return pathToNode
        for successor, action, stepCost in problem.getSuccessors(node):
            new_path = pathToNode + [action]
            total_step_cost = problem.getCostOfActions(new_path)
            heuristic_cost = total_step_cost + heuristic(successor, problem)
            prior_queue.update((successor, new_path), heuristic_cost)


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = set()
    flag = 0
    currentState = problem.getStartState()
    prior_queue = util.PriorityQueue()
    prior_queue.update((currentState, []), heuristic(currentState, problem) + 0)
    while 1:
        if prior_queue.isEmpty():
            return []
        node, pathToNode = prior_queue.pop()
        if problem.isGoalState(node):
            return pathToNode
        if node not in visited:
            visited.add(node)
            for successor, action, stepCost in problem.getSuccessors(node):
                new_path = pathToNode + [action]
                total_step_cost = problem.getCostOfActions(new_path)
                heuristic_cost = total_step_cost + heuristic(successor, problem)
                prior_queue.update((successor, new_path), heuristic_cost)
                if heuristic(node, problem) - heuristic(successor, problem) > stepCost:
                    flag = 1
            if flag == 1: # Aimed at not expending repeat nodes
                return aStarWithoutConsistency(problem, heuristic, prior_queue)
```

For a search problem, it defines a class to store all information including `getStartState(self)`, `isGoalState(self, state: Any)`, `getSuccessors(self, state: Any)`.

`getStartState(self)`: stores minimum state spaces representation

`isGoalState(self, state: Any)`: return True or False depending on the affiliated state space 

`getSuccessors(self, state: Any)`: it can change `state[1]` or the calling function changes `state[1]`.

### Design heuristics

```python
def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.
    """
    corners = problem.corners # These are the corner coordinates
    heuristic = 0
    position = state[0]
    visited_boolean = state[1] # The order of corner is (1,1), (1,top), (right, 1), (right, top)
    corners = list(corners)
    false_corners = []

    for i in range(4):
        if not visited_boolean[i]:
            false_corners.append(corners[i])

    # At first design, the manhattan distance only calculates the length to each corner and picks a minimum heuristics.
    # When it reaches one of the corner, delete the corner and recalculate length to the rest.
    # But it cannot complete the task within 1000 expended nodes.
    # Now that it calculates the total minimum length to all corners from the current state.

    while len(false_corners) > 0:
        closest_corner = None
        distance = float('inf')
        for corner in false_corners:
            if util.manhattanDistance(position, corner) < distance:
                distance = util.manhattanDistance(position, corner)
                closest_corner = corner

        heuristic += distance
        position = closest_corner
        false_corners.remove(closest_corner)
    return heuristic

def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
	""" There are an amount of food within the maze and the task is to eat them all. """
    # In this section, we cannot use the previous answer anymore, since there exists a case that all foods lie in the line so that, in previous one, we sum up length from start to each goal which overestimates the real cost.
    position, foodGrid = state
    food_grid = foodGrid.asList()
    if not food_grid:
        return 0
    # In order to expend fewer nodes, we choose to reach the furthest node using max so that we may go through some nodes in the process. mazeDistance actually is not a heuristic function as it calculates the real distance between two nodes.
    dis = max(mazeDistance(position, food_location, problem.startingGameState) for food_location in food_grid)
    return dis

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob)) # BFS finds the shortest path to the goal as the cost of each step is 1
```

When questions asked you to find **the shortest path to the goal or the closest node**, the solution was BFS or UCS.

When you apply a real cost to be a heuristic function, it expends the most fewest nodes (no detour) but it spends the most time.



## Project 2

For design an evaluation function for state rather than actions, the depth of tree would affect the accuracy of searching for a best path. Hence, it is possible that the agent cannot plan or see further resulting in the agent keep still. 

Alpha-beta pruning request to prune its main function as the data in the first branch can pass to the rest.

## Project 3













