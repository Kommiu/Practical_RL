# qlearningAgents.py
# ------------------
## based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np
from collections import defaultdict
from math import sqrt

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate aka gamma)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
      - self.getQValue(state,action)
        which returns Q(state,action)
      - self.setQValue(state,action,value)
        which sets Q(state,action) := value
    
    !!!Important!!!
    NOTE: please avoid using self._qValues directly to make code cleaner
  """
  def __init__(self, **args):
    "We initialize agent and Q-values here."
    ReinforcementAgent.__init__(self, **args)
    self._qValues = defaultdict(lambda:defaultdict(lambda:0))
    

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
    """
    return self._qValues[state][action]

  def setQValue(self,state,action,value):
    """
      Sets the Qvalue for [state,action] to the given value
    """
    self._qValues[state][action] = value

#---------------------#start of your code#---------------------#

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.
    """
    
    possibleActions = self.getLegalActions(state)
    #If there are no lega) actions, return 0.0
    if len(possibleActions) == 0:
    	return 0.0
    "*** YOUR CODE HERE ***"
    
    value = max(self.getQValue(state, a) for a in possibleActions)

    return value
    
  def getPolicy(self, state):
    """
      Compute the best action to take in a state. 
      
    """
    possibleActions = self.getLegalActions(state)

    #If there are no legal actions, return None
    if len(possibleActions) == 0:
    	return None
    
    best_action = None

    "*** YOUR CODE HERE ***"
    best_action = max(possibleActions, key=lambda a: self.getQValue(state, a))

    return best_action

  def getAction(self, state):
    """
      Compute the action to take in the current state, including exploration.  
      
      With probability self.epsilon, we should take a random action.
      otherwise - the best policy action (self.getPolicy).

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)

    """
    
    # Pick Action
    possibleActions = self.getLegalActions(state)
    action = None
    
    #If there are no legal actions, return None
    if len(possibleActions) == 0:
    	return None

    #agent parameters:
    epsilon = self.epsilon

    "*** YOUR CODE HERE ***"
    if util.flipCoin(epsilon):
        action  = random.choice(possibleActions)
    else:
        action = self.getPolicy(state)

    return action

  def update(self, state, action, nextState, reward):
    """
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf


    """
    #agent parameters
    gamma = self.discount
    learning_rate = self.alpha
    
    "*** YOUR CODE HERE ***"
    
    reference_qvalue = self.getQValue(state, action)
    updated_qvalue = reference_qvalue*(1-learning_rate) + (self.getValue(nextState)*gamma+ reward)*learning_rate


    self.setQValue(state, action, updated_qvalue)


#---------------------#end of your code#---------------------#



class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.5,gamma=0.9,alpha=0.5, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def convertState(self,state):
      pos_x, pos_y = state.getPacmanPosition()
      new_state = [pos_x,pos_y]

      ghosts = state.getGhostPositions()
      if len(ghosts) != 0:
          ghost_x, ghost_y = min([(pos_x - x,pos_y - y) for x,y in ghosts], key=lambda p: p[0]**2+p[1]**2 )
          dist = ghost_x**2 + ghost_y**2
          cos = round(ghost_x/sqrt(dist),2)
          sin = round(ghost_y/sqrt(dist),2)
          new_state.extend([dist, cos, sin])
      new_state.extend(state.hasWall(pos_x+a,pos_y+b) for a,b in [(1,1),(-1,1),(1,-1),(-1,1)] )

      food = state.getCapsules()
      if len(food) != 0:
          food_x, food_y = min([(pos_x - x,pos_y - y) for x,y in food], key=lambda p: p[0]**2+p[1]**2 )
          dist = food_x**2 + food_y**2
          cos = round(ghost_x/sqrt(dist),2)
          sin = round(ghost_y/sqrt(dist),2)
          new_state.extend([dist, cos, sin])

      mean_x = 0
      mean_y = 0
      
      for x,y in food:
          mean_x += x
          mean_y += y
      
      mean_x /= len(food)
      mean_y /= len(food)

      new_state.extend([mean_x, mean_y])




      return tuple(new_state)

  def getQValue(self, state, action):
      state = self.convertState(state)
      return QLearningAgent.getQValue(self,state,action)
  def setQValue(self, state, action, value):
      state = self.convertState(state)
      QLearningAgent.setQValue(self,state, action,value)

      


  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action



class ApproximateQAgent(PacmanQAgent):
    pass
