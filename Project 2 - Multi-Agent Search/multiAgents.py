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
        newPos = successorGameState.getPacmanPosition()
        # newFood is 2D boolean matrix of game board with "T" signaling food
        # newFood.asList() gives list of food coordinates remaining on the board
        newFood = successorGameState.getFood()
        # in addition to scaredTimer, there is getPosition() and getDirection() for each ghostState in newGhostStates
        newGhostStates = successorGameState.getGhostStates()
        # example of newScaredTimes, when two white ghost
        # [40, 40], after eating one white ghost
        # [39, 0], where new ghost that spanwed now has 0 scaredTimes
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # evaluation function is linear weighted sum of features
        # feature lists:
        # 1. position to closest
        # 2. position to ghost (closer during ghostTime, further during normalTime)
        evaluationValue = successorGameState.getScore()
        # this pacman plays very safe, but has bigger incentive to eat scared ghost than food pellet
        WEIGHT_FOOD = 5.0
        WEIGHT_SCARED_GHOST = 10.0
        WEIGHT_GHOST = 15.0

        mDistanceToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if mDistanceToFood:
            evaluationValue += WEIGHT_FOOD / min(mDistanceToFood)
        
        for ghostState in newGhostStates:
            mDistanceToGhost = manhattanDistance(newPos, ghostState.getPosition())
            if mDistanceToGhost > 0:
                if ghostState.scaredTimer > 0:
                    evaluationValue += WEIGHT_SCARED_GHOST / mDistanceToGhost
                else:
                    evaluationValue -= WEIGHT_GHOST / mDistanceToGhost

        return evaluationValue

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
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def dispatcher(gameState: GameState, depth: int, nextAgentType: str, agentIndex: int):
            # check if state is at its terminal state
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # if next agent is max or min, call respective helper function
            if nextAgentType == 'MAX':
                return maxValue(gameState, depth, agentIndex)
            elif nextAgentType == 'MIN':
                return minValue(gameState, depth, agentIndex)
            else: # fall-back state, but should never be called
                return self.evaluationFunction(gameState)
            
        def maxValue(gameState: GameState, depth: int, agentIndex: int):
            value = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                value = max(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'MIN', 1))
            return value

        def minValue(gameState: GameState, depth: int, agentIndex: int):
            value = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # if last agent, call dispatch with max since it is Pacman turn and then increase the depth
                # depth only increase at the last agent because all ghost must move in a turn within a ply
                # and only at the last agent, depth increase, new ply begins and pacman begins turn
                if agentIndex == gameState.getNumAgents()-1:
                    value = min(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth+1, 'MAX', 0))
                else:
                    value = min(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'MIN', agentIndex+1))
            return value
        
        # entry point of getAction method
        legalActions = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")

        for action in legalActions:
            tempValue = dispatcher(gameState.generateSuccessor(0, action), 0, 'MIN', 1)
            if value < tempValue:
                value = tempValue
                move = action

        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def dispatcher(gameState: GameState, depth: int, nextAgentType: str, agentIndex: int, alpha: float, beta: float):
            # check if state is at its terminal state
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # if next agent is max or min, call respective helper function
            if nextAgentType == 'MAX':
                return maxValue(gameState, depth, agentIndex, alpha, beta)
            elif nextAgentType == 'MIN':
                return minValue(gameState, depth, agentIndex, alpha, beta)
            else: # fall-back state, but should never be called
                return self.evaluationFunction(gameState)
            
        def maxValue(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            value = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                value = max(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'MIN', 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def minValue(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            value = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # if last agent, call dispatch with max since it is Pacman turn and then increase the depth
                # depth only increase at the last agent because all ghost must move in a turn within a ply
                # and only at the last agent, depth increase, new ply begins and pacman begins turn
                if agentIndex == gameState.getNumAgents()-1:
                    value = min(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth+1, 'MAX', 0, alpha, beta))
                else:
                    value = min(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'MIN', agentIndex+1, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
        
        # entry point of getAction method
        legalActions = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")
        # alpha is MAX's best option on path to root, beta is MIN's best option
        # so, initialize them to be the opposite extreme
        alpha = float("-inf")
        beta = float("inf")

        for action in legalActions:
            tempValue = dispatcher(gameState.generateSuccessor(0, action), 0, 'MIN', 1, alpha, beta)
            if value < tempValue:
                value = tempValue
                move = action
            # the dispatch starts with pacman (MAX), so update alpha accordingly
            alpha = max(alpha, value) 

        return move

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
        def dispatcher(gameState: GameState, depth: int, nextAgentType: str, agentIndex: int):
            # check if state is at its terminal state
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # if next agent is max or exp, call respective helper function
            if nextAgentType == 'MAX':
                return maxValue(gameState, depth, agentIndex)
            elif nextAgentType == 'EXP':
                return expValue(gameState, depth, agentIndex)
            else: # fall-back state, but should never be called
                return self.evaluationFunction(gameState)
            
        def maxValue(gameState: GameState, depth: int, agentIndex: int):
            value = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)
            
            for action in legalActions:
                value = max(value, dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'EXP', 1))
            return value

        def expValue(gameState: GameState, depth: int, agentIndex: int):
            value = 0
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                if agentIndex == gameState.getNumAgents()-1:
                    value += dispatcher(gameState.generateSuccessor(agentIndex, action), depth+1, 'MAX', 0)
                else:
                    value += dispatcher(gameState.generateSuccessor(agentIndex, action), depth, 'EXP', agentIndex+1)
                # uniform probability, so divide by len(legalActions)
                value += 1/len(legalActions)
            return value
        
        # entry point of getAction method
        legalActions = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")

        for action in legalActions:
            tempValue = dispatcher(gameState.generateSuccessor(0, action), 0, 'EXP', 1)
            if value < tempValue:
                value = tempValue
                move = action

        return move

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Food, Scared Ghost and Ghost has pre determined weights to it
    Key weight is scared ghost, where it grants immense incentive to eat scared ghost for bonus points

    Based on minimum manhattan distance to food, it grants plus value to the evaluation value
    Distance to ghost is divided into two parts,
    part one is where manhattan distance to ghost is greater than 2, this is when danger or incentive is not imminent
    part two is where manhattan distance to ghost is less than or equal to 2, this is whendanger or incentive is imminent
    at part two, it tries to evade/take danger/incentive by multiplying the weights by 2
    """
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    evaluationValue = currentGameState.getScore()
    SCORE_FOOD = 5.0
    SCORE_SCARED_GHOST = 100.0
    SCORE_GHOST = 1.0

    mDistanceToFood = [manhattanDistance(currentPos, food) for food in currentFood.asList()]
    if mDistanceToFood:
        evaluationValue += SCORE_FOOD / min(mDistanceToFood)

    for ghostState in ghostStates:
        mDistanceToGhost = manhattanDistance(currentPos, ghostState.getPosition())
        if mDistanceToGhost > 2:
            if ghostState.scaredTimer > 0:
                evaluationValue += SCORE_SCARED_GHOST / mDistanceToGhost
            else:
                evaluationValue -= SCORE_GHOST / mDistanceToGhost
        if mDistanceToGhost <= 2 and mDistanceToGhost > 0:
            if ghostState.scaredTimer > 0:
                evaluationValue += (SCORE_SCARED_GHOST * 2) / mDistanceToGhost
            else:
                evaluationValue -= (SCORE_GHOST * 2) / mDistanceToGhost

    return evaluationValue

# Abbreviation
better = betterEvaluationFunction
