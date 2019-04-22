import control
from control import lqr
import numpy as np

npzfile = np.load('AB.npz')
print (npzfile.files)
B = npzfile['arr_1']
A  = npzfile['arr_0']
npzfile = np.load('QR.npz')
R = npzfile['arr_1']
Q  = npzfile['arr_0']
K, S, E = lqr(A, B, Q, R)
print (A)
print (B)
print (K)
