# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for timeStep in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                action = self.getAction(state) # Pick the best action using the policy.
                if action is not None:
                    # Get the value from the state, using the maxium utility from the best q-state.
                    new_values[state] = self.getQValue(state, action)

            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qStar=0
        TransitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
        #calculate the value of all the transitional states, and return the highest one.
        for transition in TransitionStates:
            nextState = transition[0]
            prob = transition[1]
            reward = self.mdp.getReward(state, action, nextState)
            qStar += prob*(reward+self.discount*self.getValue(nextState))
        return qStar


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:         
            actions = self.mdp.getPossibleActions(state)
            
            actionValues =[]
            for action in actions:
                actionValues.append(self.getQValue(state,action))
            #find the action with the highest returned value
            maxValue = max(actionValues)
            #extract the index of the action whcih returns the highest value
            bestIndices = [index for index in range(len(actionValues)) if actionValues[index] == maxValue]
            #return the extracted action.
            return actions[bestIndices[0]]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        frontier = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = {}
        
        for tState in states:
            #create predecessor using set, as instructed 
            predecessor = set()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next, prob in transitions:
                        #if probability not equal to 0
                        if prob:
                            #find its decedants 
                            if tState == next:
                            #collect the predecessors
                                predecessor.add(state)
            predecessors[tState] = predecessor
        
        for state in states:
            if not self.mdp.isTerminal(state):
                current = self.values[state]
                qValues = []
                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    qValues = qValues + [self.getQValue(state, action)]
                #get the Q* value
                qStar = max(qValues)
                difference = current - qStar
                
                if difference > 0:
                    difference = -difference
                #prioritize the states according to the diff between current value and Q* value
                frontier.push(state, difference)
        
        for i in range(self.iterations):
            if frontier.isEmpty():
                break
            #walk through the priority Q, update the table with q*
            s = frontier.pop()
            if not self.mdp.isTerminal(s):
                values = []
                for action in self.mdp.getPossibleActions(s):
                    value = 0
                    for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                        value = self.getQValue(s,action)
                    values.append(value)
                self.values[s] = max(values)

            for predecessor in predecessors[s]:
                current = self.values[predecessor]
                qValues = []
                #check the value of current state and compare with qStar
                for action in self.mdp.getPossibleActions(predecessor):
                    qValues += [self.getQValue(predecessor, action)]
                qStar = max(qValues)
                difference = abs((current - qStar))
                #if the difference is less than the threshold, learning is done
                if (difference > self.theta):
                    frontier.update(predecessor, -difference)
