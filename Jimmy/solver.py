import numpy as np 
from numpy.linalg import eigvalsh, eigvals, eig
from scipy.linalg import norm, lstsq
import matplotlib.pyplot as plt 


class BVP:
    def __init__(self, L, N,fvec,alpha,beta, method = "D"):
        N = int(N)
        self.N = N
        self.L = L 
        self.deltax = self.L/(self.N+1)


        Adig = -2/(self.deltax)**2*np.eye(N)
        Asub = 1/(self.deltax)**2*np.eye(N-1)
        Adig[1:,:-1]+= Asub 
        Adig[:-1,1:]+= Asub
        self.A = Adig 


        self.alpha = alpha 
        self.beta = beta 
        self.fvec = fvec 
        self.bound = np.zeros((self.N))
        self.flowv = np.array([self.fvec(self.deltax*i) for i in range(1,self.N+1)])
        self.bound[0] = -alpha/self.deltax**2
        if method=="D":
            self.bound[-1] = -self.beta/self.deltax**2
        
        if method=="VN":
            self.bound[-1] = - 2/(3*self.deltax)*self.beta 
            self.A[-1,-2:]+= 1/3*np.array([-1,4])/(self.deltax**2)
            eigen = sorted(list(eigvals(self.A)))[-3:]
            self.eigen = eigen 


    def return_y(self):
        return self.flowv + self.bound 

    def gmres_it(self, kn, A, b):
        
        Q=np.empty((self.N, kn+1))
        n1=norm(b)
        Q[:,0]=b/n1
        Hessen=np.zeros((kn+1,kn)) 


        i=0 
        val=1 
        while i<kn:
            v=A@Q[:,i]
            for j in range(i+1):
                Qj = Q[:,j] 
                Hessen[j,i] = Qj@v

                v-=np.multiply(Hessen[j,i],Qj) 
            Hessen[i+1,i]=norm(v) 
            val=Hessen[i+1,i] 
            if abs(val)<1e-16:
                i+=1
                break 
        
            else:
                Q[:,i+1] = v/Hessen[i+1,i] 
                i+=1
        
        gamma = np.hstack(([n1], np.zeros((kn,)))) 
        r = lstsq(Hessen, gamma)[0] 
        xstar=Q[:,:-1] @ r  
        residue = norm(A@xstar - b, 2) 

        return residue, xstar 

    def gen_points(self,kn,b, scy = 1, scx = 1, yval = True):
        xgrid =  [scx*self.deltax*i for i in range(self.N+2)]
        if yval:
            error, solspace = self.gmres_it(kn,A=np.copy(self.A), b=b)
            solutions = np.zeros((self.N+2))
            solutions[1:-1] = solspace 
            solutions[0] = self.alpha 
            solutions[-1] = self.beta 
            solutions = scy*solutions

            return xgrid, solutions  
        else:
            return xgrid 
    
    def genx(self):
        xgrid =  [self.deltax*i for i in range(self.N+2)] 
        return xgrid 

    def bending(self):
        cross_sec = lambda x: 10**(-3)*(3 - 2*np.cos(np.pi*x/self.L)**12)
        locA = np.copy(self.A)
        b = self.flowv + self.bound 
        error, xstar = self.gmres_it(self.N+1, A = locA, b = b)
        E = 1.9*10**11 
        combfunc = lambda x: 1/(E*cross_sec(x))
        localobject = BVP(self.L, self.N, combfunc,0,0)
        bvec = localobject.return_y()
        bvecu = np.empty((self.N))
        for i in range(self.N):
            bvecu[i] = bvec[i]*xstar[i]
        return localobject.gen_points(self.N, scy=10**3, b = bvecu)

    def Noise_gen(self):
        return np.array([1e-8 for i in range(self.N)])

    def plotter(self, type, counter = 1, gmres=True):
        if type == 'bending':
            x,y = self.bending()
            print("Deformation minima:", min(y), " (mm)")
            plt.plot(x,y, label = "Deformation")
            plt.legend(title = "Legend:", loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid()
            plt.xlabel("horizontal length" + f" $(m)$")
            plt.ylabel("Deformation " + f"(mm)")
            plt.title("Deformation for the beam problem")
        
        if type== 'vn_dr':
            x = self.genx()
            y = self.Von_Neumann_Dirichlet()
            k=0
            for j in y:
                plt.plot(x,[z/max(map(abs,j)) for z in j], label="Eigen mode: " + f"{3-k}" )
                k+=1
            plt.legend(title = "Modes", loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylabel("Wave functions " + f"$\psi$")
            plt.xlabel("Position (1/L)")
            plt.title("Wave functions for the Dirichlet-Von Neumann boundary conditions")
            print(self.eigen)
            plt.grid()
            
            plt.show()
        if type == "sho":
            x = self.genx()
            energy, y = self.Schroedinger(counter, gmres)
            k=0
            for j in y:
                plt.plot(x,list(z/max([abs(min(j)), abs(max(j))])*50 + abs(energy[k]) for z in j), label="Eigen mode: " + f"{k+1}")
                k+=1
            plt.legend(title = "Modes", loc='center left', bbox_to_anchor=(1, 0.5))

            plt.xlabel("Position (1/L)")
            plt.ylabel("Wave function " + f"$\Psi$")
            plt.title("Wave functions")


        if type == "prob":
            x = self.genx()
            energy, y = self.Schroedinger(counter, gmres)
            k=0
            for j in y:
                plt.plot(x,list((z/max([abs(min(j)), abs(max(j))]))**2*50 + abs(energy[k]) for z in j), label="Eigen mode: " + f"{k+1}")
                k+=1
            
            plt.ylabel("Probability distribution onto energy level " + f"$|(\Psi|^2)$")
            plt.xlabel("Position (1/L)")
            plt.title("Probability distribution for the potential " + "$V(x)$")
            plt.legend(title = "Modes", loc='center left', bbox_to_anchor=(1, 0.5))

    def Von_Neumann_Dirichlet(self):
        eigenvals = self.eigen 
        ylist = []
        for i in eigenvals:
            ideal = list(self.gmres_it(kn= self.N, A = self.A - i*np.eye(self.N), b = self.Noise_gen())[1])
            ylast = 1/3*(2*self.deltax*self.beta + 4*ideal[-1] - ideal[-2])
            ylist.append([self.alpha] + ideal + [ylast] )
        return ylist 

    def Schroedinger(self,counter, gmres = True):
        Hamilton = -1/2*np.copy(self.A)
        for i in range(self.N):
            Hamilton[i,i]+=self.flowv[i]
        if gmres:
            spectrum = eigvalsh(Hamilton)[:counter]
            k=0
            ylist = []
            for i in spectrum:
                print("State:", k+1, "\nEnergy:", spectrum[k])
                x0 = self.gmres_it(self.N, Hamilton - i*np.eye(self.N), b = self.Noise_gen())[1]
                k+=1

                solution = [self.alpha] + list(x0) + [self.beta]
                ylist.append(solution)
        else:
            eigv, eigvec = eig(Hamilton)
            spectrum = sorted(eigv)[:counter]
            k=0
            ylist = []
            for i in spectrum:
                print("State:", k+1, "\nEnergy:", spectrum[k] )
                key = list(eigv).index(i)
                solution = [self.alpha] + list(eigvec[:,key]) + [self.beta]
                k+=1 
                ylist.append(solution)

        
        return spectrum, ylist        

zerofunc = lambda x: 0
potone = lambda x: 700*np.sin(np.pi*x)**2
linbar = lambda x: 700*(0.5 - abs(x-0.5))
central = lambda x: 700*(0.5 - x)**2 
gaussian = lambda x: 700*abs(np.exp(-(x - 0.5)**2) - np.exp(-0.5**2))

def semibarrier(x):
    if 1/5<=x<2/5 or 3/5<=x<4/5:
        return 700
    else:
        return 0
def tripbarrier(x):
    if 1/8<=x<2/8 or 3/8<=x<4/8 or 6/8<=x<7/8:
        return 700
    else:
        return 0 
def singbar(x):
    if 1/3<=x<2/3:
        return 700
    else:
        return 0