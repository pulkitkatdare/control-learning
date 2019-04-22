import numpy as np
from environment.model import env_model, env_new_model
from environment.box import Box

Q = 1*np.eye(5)
R = 1.0*np.eye(1)
np.savez('QR.npz', Q, R)
class environment(object):
    def __init__(self, model, init_set, Q = Q, R = R, state_dim=3, action_dim=1):
        self.model = model #model of the function
        self.init_set = init_set
        self.action_space = Box(low = -1.0, high = 1.0)
        self.Q = Q
        self.R = R
        self.state_dim  = state_dim
        print (self.state_dim)
        self.action_dim = action_dim
        self.state = self.reset()

    def reset(self, seed = 1234):
        np.random.seed(seed)
        dim = int(len(self.init_set)/2)


        #state = np.zeros((dim, 1))

        #for i in range(dim):
        #    state[i, 0] = np.random.uniform(self.init_set[i], self.init_set[dim + i])
        self.state = np.random.random((self.state_dim,1))#state
        return self.state

    def step(self, action):
        #print (action)
        next_state = self.model(self.state, action)
        reward =  -(np.dot(self.state.T, np.matmul(self.Q, self.state)) + np.dot(action.T, np.matmul(self.R, action)))
        if abs(action[0][0]) > 10e6:
            terminal = True
        else:
            terminal = False

        info = None
        self.state = next_state
        return next_state, reward, terminal, info

    def seed(self, seed = 1234):
        np.random.seed(seed)