import numpy as np
from environment.model import env_model, env_new_model
from environment.box import Box

Q = 1*np.eye(3)
R = 1*np.eye(1)

class environment(object):
    def __init__(self, model, init_set, Q = Q, R = R):
        self.model = model #model of the function
        self.init_set = init_set
        self.state = self.reset()
        self.action_space = Box(low = -1.0, high = 1.0)
        self.Q = Q
        self.R = R

    def reset(self, seed = 1234):
        np.random.seed(seed)
        dim = int(len(self.init_set)/2)


        state = np.zeros((dim, 1))

        for i in range(dim):
            state[i, 0] = np.random.uniform(self.init_set[i], self.init_set[dim + i])
        self.state = state
        return state

    def step(self, action):
        next_state = self.model(self.state, action)
        dist = np.sqrt(self.state[0]**2 + self.state[1]**2 + self.state[2]**2)
        reward = - ((1./(dist**2))*np.dot(self.state.T, np.matmul(self.Q, self.state)) + np.dot(action.T, np.matmul(self.R, action)))
        if next_state[1] > 9.0:
            terminal = False
        else:
            terminal = False
        info = None
        self.state = next_state
        return next_state, reward, terminal, info

    def seed(self, seed = 1234):
        np.random.seed(seed)