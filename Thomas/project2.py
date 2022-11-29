import numpy as np
import scipy.linalg as lin
import scipy.optimize as opt
import matplotlib.pyplot as plt


def RMSnorm(vec):
    """Returns the Root Mean Square norm of the vector vec.
    The RMS-norm is defined as:
    $$ \sqrt{ 1/(N+1) \cdot \sum_{k=0}^N vec_k^2} $$
    Args:
        vec (array-like): A vector.

    Returns:
        _type_: _description_
    """
    ret = 0
    for x in vec:
        ret += x**2
    ret /= len(vec)
    return np.sqrt(ret)

def getSmallesEigenPairs(A, n):
    """Finds the n smallest eigenvalues of A and their accompanying eigenvector.

    Args:
        A (ndarray): The Matrix from which to find the eigenpairs.
        n (int): Number of eigenpairs to return

    Returns:
        list [int, ndarray]: Pairs of eigenvalues and eigenvectors.
    """
    res = np.linalg.eig(A)
    pairs = [[res[0][i], res[1][:, i]] for i in range(len(res[0]))]
    return smallest(pairs, n)


def smallest(list_to_sort, n):
    """Recursively finds and sorts n elements of a list according to the smallest absolute value of the first sub-element.

    Args:
        list_to_sort ([float, ...]): List containing other lists where the first element is float.
        n (int): Number of elements to return

    Returns:
        list ([float, ... ]): Same type as list_to_sort
    """
    if len(list_to_sort) <= n:
        return sortEigenPairs(list_to_sort)
    else:
        k = len(list_to_sort)//2
        lower = smallest(list_to_sort[:k], n)
        upper = smallest(list_to_sort[k:], n)
        new_vec = []
        while len(new_vec) < n:
            if np.abs(lower[0][0]) < np.abs(upper[0][0]):
                new_vec.append(lower[0])
                if len(lower) > 1:
                    lower = lower[1:]
                else:
                    while len(new_vec) < n:
                        new_vec.append(upper[0])
                        upper = upper[1:]
                    break
            else:
                new_vec.append(upper[0])
                if len(upper) > 1:
                    upper = upper[1:]
                else:
                    while len(new_vec) < n:
                        new_vec.append(lower[0])
                        lower = lower[1:]
                    break
        return new_vec


def sortEigenPairs(list_to_sort):
    """A divide and conquer sorter that recursively sorts a list according to the smallest absolute value of the first sub-element.

    Args:
        list_to_sort ([float, ...]): List containing other lists where the first element is float.

    Returns:
        list ([float, ... ]): Same type as list_to_sort
    """
    if len(list_to_sort) == 1:
        return list_to_sort
    else:
        k = len(list_to_sort)//2
        lower = sortEigenPairs(list_to_sort[:k])
        upper = sortEigenPairs(list_to_sort[k:])
        new_vec = []
        while True:
            if np.abs(lower[0][0]) < np.abs(upper[0][0]):
                new_vec.append(lower[0])
                if len(lower) > 1:
                    lower = lower[1:]
                else:
                    for e in upper:
                        new_vec.append(e)
                    break
            else:
                new_vec.append(upper[0])
                if len(upper) > 1:
                    upper = upper[1:]
                else:
                    for e in lower:
                        new_vec.append(e)
                    break
        return new_vec


class BVP:
    """A class for Boundary Value Problems
    """
    def __init__(self, alpha, beta, L):
        """Generates a Boundary Value Problems on the interval [0,L] with boundary values alpha and beta.

        Args:
            alpha (float): Lower boundary value
            beta (float): Upper boundary value
            L (float): Upper boundary
        """
        self.alpha = alpha
        self.beta = beta
        self.L = L


class BVP_solver:
    """A class for solvers of Boundary Value Problems.
    """
    def __init__(self, N):
        """Generates a solver for Boundary Value Problems with N interior points.

        Args:
            N (int): Number of interior points.
        """
        self.N = N

    def solve(self):
        pass


class doubleDirichlet(BVP_solver):
    """A solver for Boundary Value Problems with Dirichlet conditions at both alpha and beta.

    Args:
        BVP_solver (class): Implements the BVP_solver class.
    """
    def solve(self, problem, fvec):
        """Solves the BVP and returns the solution.

        Args:
            problem (BVP): The problem to be solved
            fvec (ndarray): f evaluated at N interior points.

        Returns:
            [float]: The solution y.
        """
        h_squared = (problem.L/(self.N+1))**2
        v = np.zeros(self.N)
        v[0] = -2/h_squared
        v[1] = 1/h_squared
        A = lin.toeplitz(v)
        RHS = fvec
        RHS[0] -= problem.alpha/h_squared
        RHS[-1] -= problem.beta/h_squared
        sol = np.linalg.lstsq(A, RHS)[0]
        return [problem.alpha] + sol.tolist() + [problem.beta]


class dirichletNeumann(BVP_solver):
    """A solver for Boundary Value Problems with Dirichlet condition at both alpha and Neumann condition at beta.

    Args:
        BVP_solver (class): Implements the BVP_solver class.
    """
    def solve(self, problem, fvec):
        """Solves the BVP and returns the solution.

        Args:
            problem (BVP): The problem to be solved
            fvec (ndarray): f evaluated at N interior points.

        Returns:
            [float]: The solution y.
        """
        h_squared = (problem.L/(self.N))**2
        A = (-2/h_squared)*np.eye(self.N) + (1/h_squared)*np.eye(self.N, k=1) + (1/h_squared)*np.eye(self.N, k=-1)
        A[-1, -2] += 1/h_squared
        RHS = fvec
        RHS[0] -= problem.alpha/h_squared
        RHS[-1] -= (2*problem.beta)/h_squared
        sol = np.linalg.lstsq(A, RHS)[0]
        return [problem.alpha] + sol.tolist() + [sol[-1]+problem.beta]


class SturmLiou_DoubleDirichlet(BVP_solver):
    """A solver for Sturm-Liouville eigenvalue problems with Dirichlet conditions at both alpha and beta. The standard formulation of a S-L eigenvalue problem is
    $$ \frac{d}{dx} \left( p(x) \frac{dy}{dx}\right) - q(x)y = \lambda y.$$

    Args:
        BVP_solver (class): Implements the BVP_solver class.
    """
    def solve(self, problem, number_of_modes, p=None, q=lambda x: 0):
        """Solves the eigenvalue problem and returns a number of eigenvalues and eigenmodes.

        Args:
            problem (BVP): The problem to be solved
            number_of_modes (int): The number of eigenvalues and eigenmodes to solve for.
            p (callable, optional): The function p(x). Must be positive on [0,L]. Defaults to 1.
            q (callable, optional): The function q(x). Must be non-negative on [0,L]. Defaults to 0.

        Returns:
            ([float], [[float]]): A list of eigenvalues, and a list of eigenmodes, each being a list of evaluations of y.
        """
        if p is None:
            h = problem.L/(self.N+1)
            h_squared = (problem.L/(self.N+1))**2
            v = np.zeros(self.N)
            v[0] = -2/h_squared
            v[1] = 1/h_squared
            A = lin.toeplitz(v)

        else:
            h = problem.L/(self.N+1)
            h_squared = h**2
            A = np.zeros((self.N, self.N))
            xi = 0
            A[0, 0] -= p(xi+h/2)/h_squared
            for i in range(0, self.N-1):
                xi += h
                p1 = p(xi+h/2)/h_squared
                A[i, i] += q(xi) - p1
                A[i, i+1] += p1
                A[i+1, i] += p1
                A[i+1, i+1] -= p1
            xi += h
            A[-1, -1] += q(xi) - p(xi+h/2)/h_squared

        eigenpairs = getSmallesEigenPairs(A, number_of_modes)
        eigenmodes = []
        for ep in eigenpairs:
            eigenval, eigenvector = ep
            y_list = [problem.alpha/eigenval]
            for y in eigenvector:
                y_list.append(y)
            y_list.append(problem.beta/eigenval)
            eigenmodes.append(np.array(y_list))
        return [ep[0] for ep in eigenpairs], eigenmodes


class SturmLiou_DirichletNeumann(BVP_solver):
    """A solver for Sturm-Liouville eigenvalue problems with Dirichlet condition at alpha and Neumann condition at beta.

    Args:
        BVP_solver (class): Implements the BVP_solver class.
    """
    def solve(self, problem, number_of_modes):
        """Solves the eigenvalue problem and returns a number of eigenvalues and eigenmodes.

        Args:
            problem (BVP): The problem to be solved
            number_of_modes (int): The number of eigenvalues and eigenmodes to solve for.

        Returns:
            ([float], [[float]]): A list of eigenvalues, and a list of eigenmodes, each being a list of evaluations of y.
        """
        h_squared = (problem.L/(self.N))**2
        A = (-2)*np.eye(self.N) + np.eye(self.N, k=1) + np.eye(self.N, k=-1)
        A[-1, -2] += 1

        eigenpairs = getSmallesEigenPairs(A, number_of_modes)
        eigenmodes = []
        for ep in eigenpairs:
            eigenval, eigenvector = ep
            eigenval = eigenval/h_squared
            y_list = [problem.alpha/eigenval]
            for y in eigenvector:
                y_list.append(y)
            y_list.append(y_list[-1] + problem.beta/eigenval)
            eigenmodes.append(np.array(y_list))

        return [ep[0]/h_squared for ep in eigenpairs], eigenmodes


if __name__ == "__main__":
    def doTask1point1():
        N = 999
        L = 1
        problem = BVP(alpha=0, beta=1/2, L=L)
        solver = doubleDirichlet(N)
        numerical_solution = solver.solve(problem=problem, fvec=np.ones(N))
        x_list = np.linspace(0, L, N+2)
        actual_solution = [(x**2)/2 for x in x_list]

        plt.plot(x_list, numerical_solution, label="Numerical solution")
        plt.plot(x_list, actual_solution, label="Analytic solution")
        plt.title("$f(x) \\equiv 1$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.grid()
        plt.legend()
        plt.show()

        plt.plot(x_list, [np.abs(numerical_solution[k] - actual_solution[k]) for k in range(N+2)])
        plt.title("Error of numerical solution for $f(x) \\equiv 1$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.grid()
        plt.legend()
        plt.show()

    def doTask1point1_2():
        N_list = []
        error_list = []
        for N in range(10,499):
            N_list.append(N)
            L = 1
            problem = BVP(alpha=1, beta=np.exp(-L), L=L)
            solver = doubleDirichlet(N)
            x_list = np.linspace(0, L, N+2)
            numerical_solution = solver.solve(problem=problem, fvec=[np.exp(-x) for x in x_list[1:-1]])
            actual_solution = [np.exp(-x) for x in x_list]
            error = np.array(numerical_solution[:-1])-np.array(actual_solution[:-1])
            error_list.append(RMSnorm(error))

        plt.loglog(N_list, error_list, '.', label="$||e||_{RMS}$")
        coeff = opt.curve_fit(lambda x,c: c*(1/x)**2, N_list , error_list)[0][0]
        plt.loglog(N_list, [coeff*(1/N)**2 for N in N_list], label=f"${coeff}(1/N)^2$")
        plt.title("The RMS-norm of the error of the approximation to $f(x)=e^{-x}$")
        plt.xlabel("Number of inner points, $N$")
        plt.ylabel("")
        plt.grid()
        plt.legend()
        plt.show()

    def doTask1point2():
        N = 999
        Length = 10
        Elasticity = 1.9 * 10**11

        def Inertia(x):
            return 10**(-3) * (3-2*np.cos((np.pi*x)/Length)**12)

        problem = BVP(0, 0, Length)
        solver = doubleDirichlet(N)
        fvec1 = -50000*np.ones(N)
        Moment = solver.solve(problem, fvec1)
        x_list = np.linspace(0, Length, N+2)
        fvec2 = [Moment[i]/(Elasticity * Inertia(x_list[i])) for i in range(1, N+1)]
        u = solver.solve(problem, fvec2)

        print(f"Mid-point value: {u[len(u)//2]}")
        plt.figure(figsize=(10, 6))
        plt.plot(x_list, u)
        title_string = f"Beam deflection for $N = {N}$"
        title_string += f"\nLength = {Length} m, Elasticity = {Elasticity/10**11}"+"$\\cdot 10^{11}$ N/m$^2$"
        plt.title(title_string)
        plt.xlabel("Position (m)")
        plt.ylabel("Deflection (m)")
        plt.grid()
        plt.show()

    def do_2point1_error_compare(m, maxN):
        errors = [[] for i in range(m)]
        N_list = []
        problem = BVP(0, 0, 1)
        exact_eigs = [-((k+1/2) * np.pi)**2 for k in range(0, m+1)]
        for N in range(m, maxN):
            solver = SturmLiou_DirichletNeumann(N)
            num_eigvals, eigenmodes = solver.solve(problem, m)
            for i in range(m):
                errors[i].append(np.abs(num_eigvals[i]-exact_eigs[i]))
            N_list.append(N)
        plt.figure(figsize=(10, 6))
        for i in range(m):
            plt.loglog(N_list, errors[i], '.', label=f"$\\lambda_{i+1}$")
        plt.title("The error in approximated eigenvalues as a function of number of interior points in a loglog plot")
        plt.ylabel("Error, $|\\lambda_{\\Delta x}-\\lambda|$")
        plt.xlabel("Number of interior points, $N$")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.subplots_adjust(right=0.8)
        plt.show()

    def do_2point1_graph(k):
        N = 499
        problem = BVP(0, 0, 1)
        solver = SturmLiou_DirichletNeumann(N)
        num_eigvals, num_eigmodes = solver.solve(problem, k)
        x_list = np.linspace(0, 1, N+2)
        plt.figure(figsize=(10, 6))
        for i in range(len(num_eigmodes)):
            y = num_eigmodes[i]
            plt.plot(x_list, y, label=f"Eigenmode {i+1}")

        plt.title("Eigenmodes for $u'' = \\lambda u$ with $u(0)=u'(1)=0$.")
        plt.ylabel("$u$")
        plt.xlabel("$x$")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.subplots_adjust(right=0.8)
        plt.show()

    def shroed(V=None, p=lambda x: 1, Vformula="$V(x)=1$"):
        N = 200
        problem = BVP(0, 0, 1)
        solver = SturmLiou_DoubleDirichlet(N)
        num_eigvals, num_eigmodes = solver.solve(problem, 10, p=p, q=V)

        titles = [f"Wave function\n{Vformula}",
                  f"Probability density\n{Vformula}",
                  f"Normalized wave function\n{Vformula}",
                  f"Normalized probability density\n{Vformula}"]

        y_labels = ["Wave function, $\\Psi$",
                    "Probability density, $|\\Psi|^2$",
                    "Wave function, $\\Psi$",
                    "Probability density, $|\\Psi|^2$"]

        C = 50
        plot_funcs = [lambda y, e: y,
                      lambda y, e: np.abs(y)**2,
                      lambda y, e: e + C*y/y[np.argmax([np.abs(y)])],
                      lambda y, e: e + C*np.abs(y/np.abs(y[np.argmax([np.abs(y)])]))**2]

        x_list = np.linspace(0, 1, N+2)
        for k in range(len(titles)):
            plot_title = titles[k]
            y_label = y_labels[k]
            plot_func = plot_funcs[k]
            plt.figure(figsize=(10, 6))
            for i in range(len(num_eigmodes)):
                y = num_eigmodes[i]
                plt.plot(x_list, plot_func(y, num_eigvals[i]), label=f"Eigenmode {i+1}")
            plt.title(plot_title)
            plt.xlabel("Position, $1/L$")
            plt.ylabel(y_label)
            plt.grid()
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.subplots_adjust(right=0.8)
            plt.show()

    print("\nDo Task 1.1?")
    while True:
        flag = input("(Yes/No): ")
        flag = flag.lower()
        if flag == "y":
            print("\nf=1?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    doTask1point1()
                    break
                elif flag == "n":
                    break
            print("\nf=e^{-x}?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    doTask1point1_2()
                    break
                elif flag == "n":
                    break
            break
        elif flag == "n":
            break
    print("\nDo Task 1.2?")
    while True:
        flag = input("(Yes/No): ")
        flag = flag.lower()
        if flag == "y":
            doTask1point2()
            break
        elif flag == "n":
            break
    print("\nDo Task 2.1?")
    while True:
        flag = input("(Yes/No): ")
        flag = flag.lower()
        if flag == "y":
            print("\nCompare the errors?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    do_2point1_error_compare(m=3, maxN=100)
                    break
                elif flag == "n":
                    break
            print("\nPlot the eigenmodes?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    do_2point1_graph(3)
                    break
                elif flag == "n":
                    break
            break
        elif flag == "n":
            break
    print("\nDo Task 2.2?")
    while True:
        flag = input("(Yes/No): ")
        flag = flag.lower()
        if flag == "y":
            print("\nDo V(x)=0?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    shroed(p=lambda y: -1/2, V=lambda y: 0)
                    break
                elif flag == "n":
                    break
            print("\nDo V(x)=700(0.5 - |x-0.5|)?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    shroed(p=lambda y: -1/2, V=lambda x: 700*(0.5 - np.abs(x-0.5)), Vformula="$V(x)=700(0.5 - |x-0.5|)$")
                    break
                elif flag == "n":
                    break
            print("\nDo V(x)=800 sin^2 (pi*x)?")
            while True:
                flag = input("(Yes/No): ")
                flag = flag.lower()
                if flag == "y":
                    shroed(p=lambda y: -1/2, V=lambda x: 800*(np.sin(np.pi*x)**2), Vformula="$V(x)=800 sin^2 \\pi x$")
                    break
                elif flag == "n":
                    break
            break
        elif flag == "n":
            break
