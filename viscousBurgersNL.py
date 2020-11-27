import paraExpNL as pe
import numpy as np
import sys
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import interpolate, integrate, optimize

def Print(str):
    # For parallel printing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if(rank==0):
        print(str)
    sys.stdout.flush()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Print('Running the viscous Burgers (parallel) example')
    # Setup the problem ##
    n = 128
    dx = 2*np.pi/n
    x = np.linspace(0,2*np.pi,n+1)
    x = x[:-1]
    dx = x[1]-x[0]
    y = np.sin(x)

    # Setup the diffusion matrix
    D = 1
    dm1 = D/(dx**2)
    d   = -2*D/(dx**2)
    dp1 = D/(dx**2)

    diagonals = [dm1, d, dp1]
    offsets   = [-1, 0, 1]

    A = sp.diags(diagonals, offsets, (n, n))
    A = sp.csr_matrix(A)
    # Set the periodic BCs
    A[0,-1] = dm1
    A[-1,0] = dp1

    dm1 = -1/(2*dx)
    dp1 =  1/(2*dx)

    diagonals = [dm1, dp1]
    offsets   = [-1, 1]

    Dx = sp.diags(diagonals, offsets, (n, n))
    Dx = sp.csr_matrix(Dx)
    # Set the periodic BCs
    Dx[0,-1] = dm1
    Dx[-1,0] = dp1

    T = 10
    ######################

    hompartDir    = lambda q : A @ q
    hompartAdj    = lambda qadj,qdir : A.T @ qadj -qdir*(Dx.T@qadj)-qadj*(Dx.T@qdir)
    nonlinPart = lambda q : -q*(Dx@q)
    inhomPart  = lambda t, f : f*np.sin(t)

    # Generate the true solution
    fTrue = np.sin(x)
    equ = lambda t, q : A@q +nonlinPart(q)+ inhomPart(t,fTrue)
    Print("## Generating the true solution ##\n")
    solTrue = integrate.solve_ivp( equ, (0,T), np.zeros(n), method='RK45')
    qTrue = interpolate.interp1d(solTrue.t,solTrue.y,bounds_error=False,fill_value=(solTrue.y[:,0],solTrue.y[:,-1]))

    costIntegranddq = lambda t , q : 2*(q-qTrue(t))
    equationdf      = lambda t , q : np.sin(t)
    costIntegrand   = lambda t , q : np.linalg.norm(q-qTrue(t))**2

    solCtx = pe.paraExpIntegrator(T,hompartDir, nonlinPart,hompartAdj,costIntegrand,costIntegranddq,equationdf,inhomPart,n)
    #
    Print("\n## Running the optimisation ##\n")
    f0 = np.sin(2*x) + np.cos(4*x)
    if(rank==0):
        opts = {'disp': True}
    else:
        opts = {}
    comm.barrier()
    optSol = optimize.minimize(solCtx.directAdjointLoop,f0,method='L-BFGS-B',tol=1e-3,jac=True,options=opts)

    if(rank==0):
        plt.plot(x,f0)
        plt.plot(x,fTrue,'x')
        plt.plot(x,optSol.x)

        plt.xlabel(r'$x$')
        plt.ylabel(r'$q(x)$')
        plt.legend(('Initial guess','True solution','Optimisation solution'))
        plt.show()
