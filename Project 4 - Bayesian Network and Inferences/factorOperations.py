# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    unconditionedVariableSet = set()
    conditionedVariableSet = set()
    variableDomainDict = {}

    # for each factor, place them in unconditioned and conditioned sets
    for factor in factors:
        # as given in the problem, variableDomainsDict are all same, so set it once
        if not variableDomainDict:
            variableDomainDict = factor.variableDomainsDict()
        for uc in factor.unconditionedVariables():
            unconditionedVariableSet.add(uc)
        for c in factor.conditionedVariables():
            conditionedVariableSet.add(c)

    # remove any conditioned that also exist in unconditioned (This is "Joining")
    conditionedVariableSet = conditionedVariableSet - unconditionedVariableSet

    joinedFactor = Factor(list(unconditionedVariableSet), list(conditionedVariableSet), variableDomainDict)

    # One example of two factors from the test case, P(D|W) * P(W) = P(D,W)
    # P(D | W)
    # |  D  |  W   |  Prob:  |
    # ------------------------
    # | wet | sun  | 0.1000000 |
    # | dry | sun  | 0.9000000 |
    # ------------------------
    # | wet | rain | 0.7000000 |
    # | dry | rain | 0.3000000 |
    # P(W)
    # |  W   |  Prob:  |
    # ------------------
    # | sun  | 0.8000000 |
    # | rain | 0.2000000 |

    # Below loop calculates joint probability resulting from product rule
    # {'W': 'sun', 'D': 'wet'} 0.1
    # {'W': 'sun', 'D': 'wet'} 0.8 --> result should be P(D=wet, W=sun) = .08
    # ----
    # {'W': 'sun', 'D': 'dry'} 0.9
    # {'W': 'sun', 'D': 'dry'} 0.8 --> result should be P(D=dry, W=sun) = .72
    # ----
    # {'W': 'rain', 'D': 'wet'} 0.7 
    # {'W': 'rain', 'D': 'wet'} 0.2 --> result should be P(D=wet, W=rain) = .14
    # ----
    # {'W': 'rain', 'D': 'dry'} 0.3
    # {'W': 'rain', 'D': 'dry'} 0.2 --> result should be P(D=dry, W=rain) = .06
    # .08 + .72 + .14 + .06 = 1 as well
    
    for assignment in joinedFactor.getAllPossibleAssignmentDicts():
        probability = 1
        for factor in factors:
            probability *= factor.getProbability(assignment)
        
        joinedFactor.setProbability(assignment, probability)

    return joinedFactor

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))
        
        unconditionedVariableList = list(factor.unconditionedVariables())
        conditionedVariableList = list(factor.conditionedVariables())
        # Factor stores variableDOmainsDict of original BayesNet, not only the variables they use
        # so, there is no need to remove eliminationVariable key from this Dict, inputDict == outputDict
        variableDomainDict = factor.variableDomainsDict()
        unconditionedVariableList.remove(eliminationVariable)

        # simple example from first test case
        # P(D, W)
        # |  D  |  W   |  Prob:  |
        # ------------------------
        # | wet | sun  | 8.0e-02 |
        # | dry | sun  | 0.7200000 |
        # | wet | rain | 0.1400000 |
        # | dry | rain | 6.0e-02 |
        # elimination variable is W, should result in P(D=wet)=.22, P(D=dry)=.78
        # variableDomainDict = {'W': ['sun', 'rain'], 'D': ['wet', 'dry']}

        eliminatedFactor = Factor(unconditionedVariableList, conditionedVariableList, variableDomainDict)

        # since we eliminated variable from factor already, we have to create new assignment
        # and then grab the probability and then combine onto existing assignment
        # existing assignments are { D : wet } and { D : dry }
        # for every variableDomainDict[eliminationVariable] = [sun,rain], we create new assignment as seen below
        # -- when D : wet --
        # {'D': 'wet', 'W': 'sun'} 0.08
        # {'D': 'wet', 'W': 'rain'} 0.14
        # -- when D : dry --
        # {'D': 'dry', 'W': 'sun'} 0.72
        # {'D': 'dry', 'W': 'rain'} 0.06
        # and simple add these probability according to D:wet or D:dry
        for assignment in eliminatedFactor.getAllPossibleAssignmentDicts():
            probability = 0
            for ev in variableDomainDict[eliminationVariable]:
                combineAssignment = assignment.copy()
                combineAssignment[eliminationVariable] = ev
                probability += factor.getProbability(combineAssignment)
            
            eliminatedFactor.setProbability(assignment, probability)
        
        return eliminatedFactor
    
    return eliminate

eliminate = eliminateWithCallTracking()