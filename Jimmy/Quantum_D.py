from solver import *

classobject = BVP(1,499,semibarrier,0,0,"D")
classobject.plotter("prob", 10, False)
plt.grid()
plt.show()