import random
import numpy
import copy

class Agent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        
    def policy(self, state):
        pass
        
    def update(self, state, action, reward, next_state):
        pass
        

class RandomAgent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))
        
    def update(self, state, action, reward, next_state):
        pass
        
class QLearningAgent:

    def __init__(self, nr_actions, discount_factor, learning_rate, epsilon_decay, min_epsilon=.01):
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.nr_actions = nr_actions
        self.Q_table = {} # dictionary!!!
        
    def Q_values(self, state):
        if state not in self.Q_table:
            return numpy.zeros(self.nr_actions)
        else:
            return self.Q_table[state]
        
    def policy(self, state):
        # Select action according to epsilon-greedy strategy
        if numpy.random.rand() < self.epsilon:
            return random.choice(range(self.nr_actions))
        else:
            Q_values = self.Q_values(state)
            return numpy.argmax(Q_values)
        
    def update(self, state, action, reward, next_state):
        old_Q_value = self.Q_values(state)[action]
        new_Q_value = reward + self.discount_factor*max(self.Q_values(next_state))
        self.Q_table[state] = self.Q_values(state) # Lazy initialization of Q-values
        self.Q_table[state][action] = (1-self.learning_rate)*old_Q_value + self.learning_rate*new_Q_value
        self.epsilon = max(self.min_epsilon, self.epsilon-self.epsilon_decay)
        

class MonteCarloPlanningAgent:

    def __init__(self, nr_actions, env, discount_factor, horizon, simulations):
        self.nr_actions = nr_actions
        self.env = env
        self.discount_factor = discount_factor
        self.horizon = horizon
        self.simulations = simulations
        
    def policy(self, state):
        Q_values = numpy.zeros(self.nr_actions)
        action_counts = numpy.zeros(self.nr_actions)
        for _ in range(self.simulations):
            generative_model = copy.deepcopy(self.env)
            # 1. Generate random plan of length h=horizon
            plan = numpy.random.randint(0, self.nr_actions, self.horizon)
            discounted_return = 0
            # 2. Simulate plan with generative model
            for t,action in enumerate(plan):
                _, reward, _, _ = generative_model.step(action)
                discounted_return += reward*(self.discount_factor**t)
            # 3. Update Q-value estimate of first action in plan
            action = plan[0]
            old_Q = Q_values[action]
            new_Q = action_counts[action]*old_Q + discounted_return
            new_Q /= action_counts[action] + 1
            Q_values[action] = new_Q
            action_counts[action] += 1
        return numpy.argmax(Q_values)
        
    def update(self, state, action, reward, next_state):
        pass   

class PlanningAndLearningAgent:

    def __init__(self, nr_actions, env, discount_factor, horizon, simulations, warmup_phase):
        self.nr_actions = nr_actions
        self.env = env
        self.discount_factor = discount_factor
        self.horizon = horizon
        self.simulations = simulations
        self.Q_learner = QLearningAgent(nr_actions, discount_factor, learning_rate=0.1, epsilon_decay=0.0001, min_epsilon=0.01)
        self.step = 0
        self.warmup_phase = warmup_phase
        
    def state_value(self, state):
        return max(self.Q_learner.Q_values(state))
        
    def policy(self, state):
        self.step += 1
        if self.step < self.warmup_phase:
            return random.choice(range(self.nr_actions))
        Q_values = numpy.zeros(self.nr_actions)
        action_counts = numpy.zeros(self.nr_actions)
        for _ in range(self.simulations):
            generative_model = copy.deepcopy(self.env)
            # 1. Generate random plan of length h=horizon
            plan = numpy.random.randint(0, self.nr_actions, self.horizon)
            discounted_return = 0
            # 2. Simulate plan with generative model
            for t,action in enumerate(plan):
                state, reward, _, _ = generative_model.step(action)
                discounted_return += reward*(self.discount_factor**t)
            # 3. Evaluate leaf state with value function
            discounted_return += self.state_value(state)*(self.discount_factor**self.horizon)
            # 4. Update Q-value estimate of first action in plan
            action = plan[0]
            old_Q = Q_values[action]
            new_Q = action_counts[action]*old_Q + discounted_return
            new_Q /= action_counts[action] + 1
            Q_values[action] = new_Q
            action_counts[action] += 1
        return numpy.argmax(Q_values)
        
    def update(self, state, action, reward, next_state):
        self.Q_learner.update(state, action, reward, next_state)
        
        
        