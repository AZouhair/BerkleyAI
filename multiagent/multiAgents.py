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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        food = newFood.asList()
        if food:
            distFood = min([manhattanDistance(newPos, dot) for dot in food])
        else:
            distFood = 0
        
        distGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if distGhost != 0:
            return -distFood - 2/distGhost - (20*len(food))
        else:
            return -9999 


def scoreEvaluationFunction(currentGameState):
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
    
    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"
        def maxValue(gameState,depth):
            Actions=gameState.getLegalActions(0)
            if not Actions or gameState.isWin() or gameState.isLose() or depth>=self.depth:            
                return(self.evaluationFunction(gameState),None)
            actionList =[(-9999,None)]
            for action in Actions:    
                actionList.append((minValue(gameState.generateSuccessor(0,action),1,depth)[0],action))                                                      
            return max(actionList)

        def minValue(gameState,agentInd,depth):
            Actions=gameState.getLegalActions(agentInd)
            if len(Actions) == 0:
                return(self.evaluationFunction(gameState),None)
            actionList = [(9999,None)]
            for action in Actions:
                if(agentInd==gameState.getNumAgents() -1):
                    actionList.append((maxValue(gameState.generateSuccessor(agentInd,action),depth + 1)[0],action))
                else:
                    actionList.append((minValue(gameState.generateSuccessor(agentInd,action),agentInd+1,depth)[0],action ))     
            return min(actionList)
        maxValue=maxValue(gameState,0)[1]
        return maxValue                                   
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState,depth,alpha,beta):
            Actions = gameState.getLegalActions(0) 
            if not Actions or gameState.isWin() or gameState.isLose() or depth>=self.depth:
                return (self.evaluationFunction(gameState), None)

            value=-9999
            Act=None
            for action in Actions:
                cost=minValue(gameState.generateSuccessor(0,action),1,depth,alpha,beta)[0]
                if value<cost:
                    value,Act=cost,action
                if value>beta:
                    return (value,Act)
                alpha = max(alpha,value)
            return (value,Act)

        def minValue(gameState,agentInd,depth,alpha,beta):
            Actions=gameState.getLegalActions(agentInd) #
            if not Actions:
                return (self.evaluationFunction(gameState),None)
            value = 9999
            Act = None
            for action in Actions:
                if (agentInd == gameState.getNumAgents() - 1):
                    cost = maxValue(gameState.generateSuccessor(agentInd,action),depth + 1,alpha,beta)[0]
                else:
                    cost = minValue(gameState.generateSuccessor(agentInd,action),agentInd + 1,depth,alpha,beta)[0]
                if (cost<value):
                    value,Act=cost,action
                if (value<alpha):
                    return (value,Act)
                beta = min(beta,value)
            return(value,Act)                                                                                      

        return maxValue(gameState,0,-9999,9999)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, agentInd, depth):
            if agentInd >= gameState.getNumAgents():
                if depth >= self.depth:
                    return self.evaluationFunction(gameState)
                else:
                    return expectimax(gameState, 0, depth + 1)
            else:
                Actions = gameState.getLegalActions(agentInd)
                if not Actions:
                    return self.evaluationFunction(gameState) 
                actionValueList = [expectimax(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth) for action in Actions]
                if agentInd == 0:
                    return max(actionValueList)
                else:
                    return sum(actionValueList) / len(Actions)

        Actions = gameState.getLegalActions(0)
        actionList=[]
        for action in Actions:
            actionList.append((expectimax(gameState.generateSuccessor(0, action), 1, 1),action))
        return max(actionList)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Same as before, we take try to minimize the distance to the food , 
    the quantity of food available and maximize the distance to the ghost, after playing with the
    weights we come to these weights. 
    We can add the average distance to the food so that pacman will try to go where there is more food density.
    I was not able to implement the search for a capsule while the ghosts aren't scared efficiently (time out).
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    food = newFood.asList()
    avg = 0

    if food:
        distFood = min([manhattanDistance(newPos, dot) for dot in food])
        avg = sum([manhattanDistance(newPos, dot) for dot in food])/len(food)
    else:
        distFood = 0
    
    distGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if distGhost != 0:
        return -distFood - 2/distGhost - (20*len(food)) + 0.1*avg 
    else:
        return -9999 

# Abbreviation
better = betterEvaluationFunction
