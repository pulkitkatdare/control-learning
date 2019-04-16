import os
import sys
import math
import copy
import torch
import random
import argparse
import numpy as np
from control import lqr
import torch.nn as nn
from time import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as weight_init


from utils.ounoise import OUNoise
from models.actor import ActorNetwork
from models.critic import CriticNetwork
from agent.ddpg import DDPGAgent
from utils.replaybuffer import ReplayBuffer
from environment.model import env_new_model
from environment.env import environment

def main(args):
	CUDA = torch.cuda.is_available()
	OUTPUT_RESULTS_DIR = './saver'
	init_set = np.array([0, 0, 1, 1, 0, 1])
	env = environment(env_new_model, init_set)
	state_dim = 3
	action_dim = 1
	action_bound = np.array([1,-1])

	actor = ActorNetwork(state_dim, action_dim, action_bound, args.actor_lr, args.tau, args.seed)
	target_actor = ActorNetwork(state_dim, action_dim, action_bound, args.actor_lr, args.tau, args.seed)
	critic = CriticNetwork(state_dim, action_dim, action_bound, args.critic_lr, args.tau, args.l2_decay, args.seed)
	target_critic = CriticNetwork(state_dim, action_dim, action_bound, args.critic_lr, args.tau, args.l2_decay, args.seed)

	if CUDA: 
		actor = actor.cuda()
		target_actor = target_actor.cuda()
		critic = critic.cuda()
		target_critic = target_critic.cuda()

	replay_buffer = ReplayBuffer(args.bufferlength, args.seed)

	agent = DDPGAgent(actor, target_actor, critic, target_critic, replay_buffer,
					  batch_size=args.batch_size, gamma = args.gamma, seed=args.seed, 
					  episode_len=args.episode_len, episode_steps=args.episode_steps,
					  noise_mean = args.noise_mean, noise_th=args.noise_th, noise_std=args.noise_std,
					  noise_decay=args.noise_decay)

	agent.train(env)
	#A = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
	#B = np.zeros((3, 1))
	#Q = np.eye(3)
	#R = np.eye(1)
	#K, S, E = lqr(A, B, Q, R)
	#print (K)
	#agent.save_actor_weights(save_dir=OUTPUT_RESULTS_DIR, filename=args.actor_weights)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='DDPG working code for classical control tasks')
	parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=1234')
	parser.add_argument('--actor_lr', type=float, default=0.0001, help='actor learning rate')
	parser.add_argument('--critic_lr', type=float, default=0.001, help='critic learning rate')
	parser.add_argument('--batch_size', type=int, default=2, help='critic learning rate')
	parser.add_argument('--bufferlength', type=float, default=20000, help='buffer size in replay buffer')
	parser.add_argument('--l2_decay', type=float, default=0.01, help='weight decay')
	parser.add_argument('--tau', type=float, default=0.001, help='adaptability')
	parser.add_argument('--gamma', type=float, default=1.00, help='discount factor')
	parser.add_argument('--episode_len', type=int, default=1000, help='episodic lengths')
	parser.add_argument('--episode_steps', type=int, default=100, help='steps per episode')
	parser.add_argument('--noise_mean', type=float, default=0.0, help='noise mean')
	parser.add_argument('--noise_th', type=float, default=0.15, help='noise theta')
	parser.add_argument('--noise_std', type=float, default=0.20, help='noise standard deviation')
	parser.add_argument('--noise_decay', type=int, default=25, help='linear decrease in noise')
	parser.add_argument('--is_train', type=bool, default=True, help='train mode or test mode. Default is test mode')
	parser.add_argument('--actor_weights', type=str, default='ddpg_pendulum', help='Filename of actor weights. Default is actor_pendulum')
	args = parser.parse_args()

	main(args)
