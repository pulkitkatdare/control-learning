import control
from control import lqr
import numpy as np
npzfile = np.load('AB.npz')
print (npzfile.files)
B = npzfile['arr_1']
A  = npzfile['arr_0']
print (A)
print (B)
#A =  np.random.randn(20, 20)
#A = A + A.T
#A = 0.001*(A + A.T)
#B = np.random.random((20, 1))
C  = control.ctrb(A, B)
print (np.linalg.matrix_rank(C))
