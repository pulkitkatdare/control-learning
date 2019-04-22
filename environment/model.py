from __future__ import division
import matplotlib
from math import *
import numpy as np
from numpy import linalg as LA 
import matplotlib.pyplot as plt
import argparse

A =  np.random.randn(5, 5)
#A = 0.001*(A + A.T)
B = np.random.random((5, 1))
#B =  np.random.random((100,1))
#B[0] = 0.3
#B[4] = 4.3
#B[7] = 3.7
#B[9] = 1.0
#B[5] = 1.0
Dt = 0.01
np.savez('AB.npz', A, B)
print ("A", A)
print ("B", B)

def diff_eqn(state, U, A = A, B = B):

	Xnewdot = np.matmul(A, state) + np.dot(B, U)
	return Xnewdot

def env_model(state, U, A = A, B = B):
	'''
	x_t+1 =  f(x_t, u_t)
	Input: 
	state_t: state of the system at time t
	delta_f: the angular of the front wheels at time t
	dt: 
	Output: 
	state_t = the next state
	'''
	#k1 = diff_eqn(state, U)
	#k2 = diff_eqn(state + (k1 / 2) * Dt, U)
	#k3 = diff_eqn(state + (k2 / 2) * Dt, U)
	#k4 = diff_eqn(state + (k3) * Dt, U)
	Xnew = diff_eqn(state, U) #state + (k1 + 2 * k2 + 2 * k3 + k4) * Dt / 6


	return Xnew

def env_new_model(state, U):

	state = env_model(state, U)
	return state + 1.0*np.random.randn(len(state),1)



def test_module():
	state = np.array([0, 0, 0])
	delta = 0.0
	state_old = env_model(state, delta)
	print (state_old)
	state_new = env_new_model(state, delta)
	print (state_new)










