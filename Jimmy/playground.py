import numpy as np 
from numpy.linalg import eig
import matplotlib.pyplot as plt

N = 1000
deltax = 1/(N+1)**2 
xgrid = [i/(N+1) for i in range(N+2)]
diag = -2/deltax*np.eye(N)
offdiag = 1/deltax*np.eye(N-1)
diag[1:,:-1]+= offdiag 
diag[:-1,1:]+= offdiag 
eigv, eigvec = eig(diag)
items = sorted(eigv)[-6:]
point = [list(eigv).index(i) for i in items]
print(point)
alpha, beta = 0,0 
for j in point:
    plt.plot(xgrid, [alpha] + list(eigvec[:,j]) + [beta])

plt.show()