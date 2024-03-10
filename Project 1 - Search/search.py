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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    # Start of DFS search implementation code
    visitedStates = set()
    dfsStack = util.Stack()
    # Pass in tuple of (stateCoordinate, movementList) onto dfsStack (fringe)
    # MovementList is movement up to the respective stateCoordinate.
    # It includes entire movementList for each state, which allows natural backtracking.
    # If pacman runs into dead-end, that entire path will be gone and pacman will go back to the split way.
    # From there, pacman will proceed to the next available path since visitedSet keeps track of visted states.
    dfsStack.push((problem.getStartState(), []))

    while(not dfsStack.isEmpty()):
        currentState, movementList = dfsStack.pop()
        visitedStates.add(currentState)
        if(problem.isGoalState(currentState)):
            return movementList
        else:
            successorList = problem.getSuccessors(currentState)
            for successorState, direction, stepCost in successorList:
                if not successorState in visitedStates:
                    # List concatenation appends two list and creates a new copy of concatenation.
                    dfsStack.push((successorState, movementList + [direction]))

    return movementList

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    visitedStates = set()
    bfsQueue = util.Queue()
    bfsQueue.push((problem.getStartState(), []))

    while(not bfsQueue.isEmpty()):
        currentState, movementList = bfsQueue.pop()
        visitedStates.add(currentState)
        if(problem.isGoalState(currentState)):
            return movementList
        else:
            successorList = problem.getSuccessors(currentState)
            for successorState, direction, stepCost in successorList:
                if not successorState in visitedStates:
                    bfsQueue.push((successorState, movementList + [direction]))
                    visitedStates.add(successorState)
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    ucsPQ = util.PriorityQueue()
    # since UCS relies on cost information, now store triplet of stateCoordinate, direction, cost
    # also for the priority item, cost will be used as well
    ucsPQ.push((problem.getStartState(), [], 0), 0)
    # visitedStates is now dictionary instead of set since it needs to keep track of cost of a state so far
    visitedStates = dict()

    while not ucsPQ.isEmpty():
        currentState, movementList, currentCost = ucsPQ.pop()
        visitedStates[currentState] = currentCost
        if(problem.isGoalState(currentState)):
            return movementList
        else:
            successorList = problem.getSuccessors(currentState)
            for successorState, direction, stepCost in successorList:
                totalCost = currentCost + stepCost
                if (not successorState in visitedStates) or (successorState in visitedStates and visitedStates[successorState] > totalCost):
                    visitedStates[successorState] = totalCost
                    ucsPQ.push((successorState, movementList + [direction], totalCost), totalCost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    aStarPQ = util.PriorityQueue()
    # priority is decided by combined cost (path cost + heuristic)
    # cost inside state triplet is only path cost, because while path cost is accumulative
    # heuristic cost is not accmulative, so we cannot include heuristic cost along with path cost
    aStarPQ.push((problem.getStartState(), [], 0), 0)
    visitedStates = dict()

    while not aStarPQ.isEmpty():
        currentState, movementList, currentCost = aStarPQ.pop()
        visitedStates[currentState] = currentCost
        if problem.isGoalState(currentState):
            return movementList
        else:
            successorList = problem.getSuccessors(currentState)
            for successorState, direction, stepCost in successorList:
                totalCost = currentCost + stepCost + heuristic(successorState, problem)
                if (not successorState in visitedStates) or (successorState in visitedStates and visitedStates[successorState] > totalCost):
                    visitedStates[successorState] = totalCost
                    # cost information inside triplet only includes path cost up to the successor state(accumulative)
                    # however priority is the total cost including path cost up to the successor state + heuristic at successor state
                    aStarPQ.push((successorState, movementList + [direction], currentCost + stepCost), totalCost)
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
