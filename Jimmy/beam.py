from solver import *


load = lambda x: -50*10**3 

classobject = BVP(10,999, load, 0, 0)
classobject.plotter('bending')

plt.show()