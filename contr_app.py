import control
from control import lqr
import numpy as np 
A = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
B = np.zeros((3, 1))
B[0] = 1.0
Q = np.eye(3)
R = np.eye(1)
K, S, E = lqr(A, B, Q, R)
print (K)
