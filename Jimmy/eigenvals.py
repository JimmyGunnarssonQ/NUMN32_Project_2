import numpy as np 
from numpy.linalg import eigvalsh, eigvals 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def delta(x,coef):
    return coef*1/(x+1)**2


class BVP:
    def __init__(self, L, N):
        N = int(N)
        self.N = N
        self.L = L 
        self.deltax = self.L/(self.N+1)


        Adig = -2/(self.deltax)**2*np.eye(N)
        Asub = 1/(self.deltax)**2*np.eye(N-1)
        Adig[1:,:-1]+= Asub 
        Adig[:-1,1:]+= Asub
        self.A = Adig 
        self.A[-1,-2:]+= 1/3*np.array([-1,4])/(self.deltax**2)
    
    def ret_A(self):
        return sorted(list(eigvals(self.A)))[-3:]
L = np.array(sorted([-(2*j-1)**2*np.pi**2/4 for j in range(1,4)]))
k=0
Nlist = [i for i in range(4,1000,25)]
eigen_log = np.empty((len(Nlist),3))
for i in Nlist:
    eigen_log[k,:] =  np.array(BVP(1,i).ret_A()) - L 
    k+=1
    print(i)

print(eigen_log)
colors = ["orange", "blue", "green"]
for i in range(3):
    y = eigen_log[:,i] 
    c,_ = curve_fit(delta, Nlist[10:], y[10:])
    plt.plot(Nlist, [delta(i,c[0]) for i in Nlist], color = colors[i])
    plt.plot(Nlist, eigen_log[:,i], "o", label=f"$\lambda \sim $" + f"{int(L[i])}", color = colors[i])

plt.yscale("log")
plt.xscale("log")
plt.ylabel("Error " + f"$\epsilon$")
plt.xlabel("Matrix size (N)")
plt.title("Eigenvalue Error versus size of matrix")
plt.grid()
plt.legend()
plt.show()
