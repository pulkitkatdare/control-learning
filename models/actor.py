import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

class ActorNetwork(nn.Module):
	def __init__(self,state_dim, action_dim, action_bound, learning_rate, tau,seed):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate
		self.tau = tau
		self.seed = seed
		
		super(ActorNetwork, self).__init__()

		self.layer1 = nn.Linear(self.state_dim,self.action_dim, bias = False)
		n = weight_init._calculate_fan_in_and_fan_out(self.layer1.weight)[0]
		#torch.manual_seed(self.seed)
		self.layer1.weight.data.normal_(0.0,math.sqrt(2./n))
		torch.manual_seed(self.seed)
		#self.layer1.bias.data.normal_(0.0,math.sqrt(2./n))

		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		
	def forward(self,x):
		y = -self.layer1(x)
		return x,y,y

	def predict(self,x):
		x,y,scaled_y = self.forward(x)
		return scaled_y

	def train(self, states, critic):
		self.optimizer.zero_grad()
		action = self.predict(states)
		output = -torch.mean(critic.predict_target(states,action))
		output.backward()
		self.optimizer.step()
		#print (self.layer1.weight)
		#self.optimizer.zero_grad()

	def update_target_network(self,target_actor):
		for weight,target_weight in zip(self.parameters(),target_actor.parameters()):
			weight.data = (1-self.tau)*weight.data + self.tau*target_weight.data
