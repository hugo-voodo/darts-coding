import random

class Agent:

    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        
    def policy(self, state):
        pass
        
    def update(self, state, action, reward, next_state):
        pass
        
def argmax(values):
    max_value = max(values)
    max_values = [i for i,e in enumerate(filter(lambda x: x == max_value, values))]
    return random.choice(max_values)
