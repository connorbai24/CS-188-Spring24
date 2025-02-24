# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() # (4, 7)
        newFood = successorGameState.getFood() #if newFood[x][y] == True:...
        newGhostStates = successorGameState.getGhostStates() # as there are more than one ghost.
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food_pos_list = newFood.asList()
        if food_pos_list:
            nearest_food = (min([manhattanDistance(newPos, food) for food in newFood.asList()]))
            agent_food = 9 / nearest_food
        else:
            agent_food = 0

        ghost_agent = float('inf')
        for ghost in newGhostStates:
            ghost_agent = min(ghost_agent, manhattanDistance(newPos, ghost.configuration.pos))
        if ghost_agent < 2:
            return float('-inf')

        scared_ghost_score = 99999999 if ghost_agent <= newScaredTimes[0] else 0
        # totalScaredTimes = sum(newScaredTimes)
        score = agent_food + successorGameState.getScore() + scared_ghost_score
        return score

        # print("Game state:", successorGameState)
        # print("new position", newPos)
        # print("new food as a list:", newFood)
        # for ghost in newGhostStates:
        #     print("new ghost state:", ghost)
        # print("scared times",newScaredTimes)
        # print("food position:", food_pos_list)
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        selfLegalMoves = gameState.getLegalActions(0) # {NORTH, SOUTH, WEST, EAST, STOP}
        best_value = -float('inf')
        best_action = None
        for action in selfLegalMoves:
            # When we evaluate the value of current agent's action, we have to observe other agents' reaction.
            successor = self.getValue(gameState.generateSuccessor(0, action), 1, 0)
            if best_value < successor:
                best_value = successor
                best_action = action
        return best_action

    def getValue(self, currState, agentIndex, depth):
        # if game losses and wins, it should be terminated.
        if currState.isWin() or currState.isLose():
            return self.evaluationFunction(currState)

        if agentIndex == 0:
            depth += 1
            if depth == self.depth:
                return self.evaluationFunction(currState)
            else:
                return self.maxValue(currState, agentIndex, depth)
        else:
            return self.minValue(currState, agentIndex, depth)

    def maxValue(self, currState, agentIndex, depth):
        initial = float('-inf')
        legalMoves = currState.getLegalActions(0)
        for action in legalMoves:
            next_agent = (agentIndex + 1) % currState.getNumAgents() # make sure it recurs in the range
            initial = max(initial, self.getValue(currState.generateSuccessor(agentIndex, action), next_agent, depth))
        return initial

    def minValue(self, currState, agentIndex, depth):
        initial = float('inf')
        legalMoves = currState.getLegalActions(agentIndex)
        # It should not set all other agents here, since they might be cooperation together.
        for action in legalMoves:
            next_agent = (agentIndex + 1) % currState.getNumAgents()
            initial = min(initial, self.getValue(currState.generateSuccessor(agentIndex, action), next_agent, depth))
        return initial

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        selfLegalMoves = gameState.getLegalActions(0)
        best_value = -float('inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in selfLegalMoves:
            successor = self.getValue(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if best_value < successor:
                best_value = successor
                best_action = action
            alpha = max(alpha, best_value) # get the max value still
        return best_action

    def getValue(self, currState, agentIndex, depth, alpha, beta):
        if currState.isWin() or currState.isLose():
            return self.evaluationFunction(currState)
        if agentIndex == 0:
            depth += 1
            if depth == self.depth:
                return self.evaluationFunction(currState)
            else:
                return self.maxValue(currState, depth, alpha, beta)
        else:
            return self.minValue(currState, agentIndex, depth, alpha, beta)

    def maxValue(self, currState, depth, alpha, beta):
        initial = float('-inf')
        legalMoves = currState.getLegalActions(0)
        for action in legalMoves:
            successor = self.getValue(currState.generateSuccessor(0, action), 1, depth, alpha, beta)
            initial = max(initial, successor)
            if initial > beta: # it should not be a equal sign.
                return initial
            alpha = max(alpha, initial)
        return initial

    def minValue(self, currState, agentIndex, depth, alpha, beta):
        initial = float('inf')
        legalMoves = currState.getLegalActions(agentIndex)
        for action in legalMoves:
            next_agent = (agentIndex + 1) % currState.getNumAgents()
            successor = self.getValue(currState.generateSuccessor(agentIndex, action), next_agent, depth, alpha, beta)
            initial = min(initial, successor)
            if initial < alpha:
                return initial
            beta = min(beta, initial)
        return initial


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        selfLegalMoves = gameState.getLegalActions(0)  # {NORTH, SOUTH, WEST, EAST, STOP}
        best_value = -float('inf')
        best_action = None
        for action in selfLegalMoves:
            # When we evaluate the value of current agent's action, we have to observe other agents' reaction.
            successor = self.getValue(gameState.generateSuccessor(0, action), 1, 0)
            if best_value < successor:
                best_value = successor
                best_action = action
        return best_action

    def getValue(self, currState, agentIndex, depth):
        # if game losses and wins, it should be terminated.
        if currState.isWin() or currState.isLose():
            return self.evaluationFunction(currState)

        if agentIndex == 0:
            depth += 1
            if depth == self.depth:
                return self.evaluationFunction(currState)
            else:
                return self.maxValue(currState, agentIndex, depth)
        else:
            return self.expValue(currState, agentIndex, depth)

    def maxValue(self, currState, agentIndex, depth):
        initial = float('-inf')
        legalMoves = currState.getLegalActions(0)
        for action in legalMoves:
            next_agent = (agentIndex + 1) % currState.getNumAgents()
            initial = max(initial, self.getValue(currState.generateSuccessor(agentIndex, action), next_agent, depth))
        return initial

    def expValue(self, currState, agentIndex, depth):
        initial = 0
        legalMoves = currState.getLegalActions(agentIndex)
        for action in legalMoves:
            next_agent = (agentIndex + 1) % currState.getNumAgents()
            # For expectimax, we need to calculate the expected value of all actions.
            initial += self.getValue(currState.generateSuccessor(agentIndex, action), next_agent, depth)
        return initial

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    food_positions = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    if food_positions:
        nearest_food = (min([manhattanDistance(pacman_position, food) for food in food_positions]))
        agent_food = 9 / nearest_food
    else:
        agent_food = 0

    ghost_agent = min([manhattanDistance(pacman_position, ghost) for ghost in ghost_positions])
    if ghost_agent != 0:
        dangerGhost = - 10 / ghost_agent
    else:
        dangerGhost = 0

    totalScaredTimes = sum(ScaredTimes)

    return totalScaredTimes + dangerGhost + agent_food + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
