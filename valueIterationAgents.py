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
            maxValue = max(actionValues)
            bestIndices = [index for index in range(len(actionValues)) if actionValues[index] == maxValue]
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
        fringe = util.PriorityQueue()
        # 獲取所有的狀態
        states = self.mdp.getStates()
        # 新建一個空字典，表示前一個狀態
        predecessors = {}
        # 遍歷所有的狀態，每一個時刻t的狀態爲tSate
        for tState in states:
            # 初始化集合，集合中的元素不重複
            previous = set()
            for state in states:
                # 獲取狀態下一個可能的動作
                actions = self.mdp.getPossibleActions(state)
                # 遍歷每一個動作
                for action in actions:
                    # 獲取當前狀態下一個動作的狀態及概率
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    # 遍歷當前下一個動作到達下一個狀態、概率
                    for next, probability in transitions:
                        # 如果概率不等於0，
                        if probability != 0:
                            # t時刻的等態對於下一個時刻的狀態 ，則在前一個狀態集中加入狀態
                            if tState == next:
                                previous.add(state)
            # 給前一個狀態賦值
            predecessors[tState] = previous
        # 遍歷每一個狀態
        for state in states:
            # 如果當前狀態不是終端狀態
            if self.mdp.isTerminal(state) == False:
                # 獲取狀態的值
                current = self.values[state]
                qValues = []
                # 獲所有的狀態。
                actions = self.mdp.getPossibleActions(state)
                # 遍歷狀態中的下一個動作,計算Q值。
                for action in actions:
                    tempValue = self.computeQValueFromValues(state, action)
                    qValues = qValues + [tempValue]
                # 獲取最大的Q值。
                maxQvalue = max(qValues)
                # s的當前值與s的所有可能操作中的最高Q值之間的差的絕對值
                diff = current - maxQvalue
                # 轉爲負數
                if diff > 0:
                    diff = diff * -1
                # 將當前狀態入優先隊列
                fringe.push(state, diff)
        # 進行循環迭代
        for i in range(0, self.iterations):
            # 如果優先隊列爲空，則中斷
            if fringe.isEmpty():
                break
            # 獲取隊列中的元素
            s = fringe.pop()
            # 如果獲取的狀態不是終端狀態
            if not self.mdp.isTerminal(s):
                values = []
                # 遍歷當前狀態的下一個可能的動作
                for action in self.mdp.getPossibleActions(s):
                    value = 0
                    # 獲取當前狀態下一個動作的狀態及概率
                    for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                        # 獲取狀態下一個動作到達下一個狀態的獎勵
                        reward = self.mdp.getReward(s, action, next)
                        # 計算value值
                        value = value + (prob * (reward + (self.discount * self.values[next])))
                    values.append(value)
                # 將最大的value值作爲當前狀態的value值
                self.values[s] = max(values)

            # 遍歷當前狀態的前一個狀態列表集中的每一個狀態
            for previous in predecessors[s]:
                # 獲取前一個狀態的values值
                current = self.values[previous]
                qValues = []
                # 遍歷前一個狀態的可能的下一個動作,計算qValues
                for action in self.mdp.getPossibleActions(previous):
                    qValues += [self.computeQValueFromValues(previous, action)]
                # 獲取qValues的最大值
                maxQ = max(qValues)
                # 計算兩者的差值的絕對值
                diff = abs((current - maxQ))
                if (diff > self.theta):
                    # 更新優先隊列
                    # 如果項目已在優先級較高的隊列中，請更新其優先級並重建堆。
                    # 如果項目已經在優先級相同或較低的隊列中，則不執行任何操作。
                    # 如果項目不在優先級隊列中，請執行與self.push相同的操作。入優先隊列三元組：(priority, count, item)
                    fringe.update(previous, -diff)
