# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def genericSearch(problem: SearchProblem, data_structure, is_priority_queue=False):
    """
    通用搜索函数，可用于深度优先搜索、广度优先搜索和一致代价搜索。
    :param problem: 搜索问题实例
    :param data_structure: 使用的数据结构（栈、队列或优先队列）
    :param is_priority_queue: 是否为优先队列
    :return: 到达目标状态的动作列表
    """
    visited = set()
    currentState = problem.getStartState()
    if is_priority_queue:
        data_structure.update((currentState, []), 0)
    else:
        data_structure.push((currentState, []))

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

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # visited = set()
    # currentState = problem.getStartState()
    # stack = util.Stack()
    # stack.push((currentState, []))
    # while 1:
    #     if stack.isEmpty():
    #         return []
    #     node, pathToNode = stack.pop()
    #     visited.add(node)
    #     if problem.isGoalState(node):
    #         return pathToNode
    #     for successor, action, stepCost in problem.getSuccessors(node):
    #         if successor not in visited:
    #             new_path = pathToNode + [action]
    #             stack.push((successor, new_path))

    stack = util.Stack()
    return genericSearch(problem, stack)

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    # visited = set()
    # currentState = problem.getStartState()
    # queue = util.Queue()
    # queue.push((currentState, []))
    # while 1:
    #     if queue.isEmpty():
    #         return []
    #     node, pathToNode = queue.pop()
    #     if problem.isGoalState(node):
    #         return pathToNode
    #     if node not in visited:
    #         visited.add(node)
    #         for successor, action, stepCost in problem.getSuccessors(node):
    #             new_path = pathToNode + [action]
    #             queue.push((successor, new_path))

    queue = util.Queue()
    return genericSearch(problem, queue)

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of the least total cost first."""
    # visited = set()
    # currentState = problem.getStartState()
    # prior_queue = util.PriorityQueue()
    # prior_queue.update((currentState, []), 0)
    # while 1:
    #     if prior_queue.isEmpty():
    #         return []
    #     node, pathToNode = prior_queue.pop()
    #     if problem.isGoalState(node):
    #         return pathToNode
    #     if node not in visited:
    #         visited.add(node)
    #         for successor, action, stepCost in problem.getSuccessors(node):
    #             new_path = pathToNode + [action]
    #             cost = problem.getCostOfActions(new_path)
    #             prior_queue.update((successor, new_path), cost)

    priority_queue = util.PriorityQueue()
    return genericSearch(problem, priority_queue, True)

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


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
            if flag == 1:
                return aStarWithoutConsistency(problem, heuristic, prior_queue)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
