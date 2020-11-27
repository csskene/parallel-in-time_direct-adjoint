import exponentialIntegrators
from mpi4py import MPI
import numpy as np
import sys
from scipy import integrate, interpolate

class solStruct:
    def __init__(self,t,y):
        self.t = t
        self.y = y

class paraExpIntegrator:
    def __init__(self,T,A,costIntegrand,costIntegranddq,equationdf,inhomPart):
        self.T = T
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.tMin = rank*self.T/size
        self.tMax = (rank+1)*self.T/size
        self.A = A
        if(rank==0):
            print('## Time partition for the inhomogeneous equations ##')
        print('rank =',rank,' : ', 't∈[',self.tMin,',',self.tMax,']')
        self.inhomPart = inhomPart
        self.costIntegrand = costIntegrand
        self.costIntegranddq = costIntegranddq
        self.equationdf = equationdf
        self.y0 = np.zeros(self.A.shape[0])

    def costFunctional(self,solDir):
        costIntegrand = np.array([self.costIntegrand(t,q) for t,q in zip(solDir.t,solDir.y.T)])
        cost = np.trapz(costIntegrand,solDir.t,axis=0)
        return cost

    def calculateGradient(self,solAdj):
        gradientIntegrand = np.array([qAdj*self.equationdf(t,self.y0) for t,qAdj in zip(solAdj.t,solAdj.y.T)])
        grad = np.trapz(gradientIntegrand,solAdj.t,axis=0)
        return grad

    def directAdjointLoop(self,f):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if(size==1):
            ## Direct ##
            dirInhom = lambda t , qDir : self.A@qDir + self.inhomPart(t,f)
            solDir = integrate.solve_ivp( dirInhom, (self.tMin,self.tMax), self.y0, method='RK45')
            cost   = self.costFunctional(solDir)
            qDir   = interpolate.interp1d(solDir.t,solDir.y)

            ## Adjoint ##
            adjInhom = lambda t , qAdj : self.A.T@qAdj + self.costIntegranddq(self.T-t,qDir(self.T-t))
            solAdj = integrate.solve_ivp( adjInhom, (self.tMin,self.tMax), self.y0, method='RK45')

            # Change from τ back to t and reverse order
            solAdj.t = self.T-solAdj.t
            solAdj.t = solAdj.t[::-1]
            solAdj.y = solAdj.y[:,::-1]

            grad   = self.calculateGradient(solAdj)
        else:
            ## Direct ##

            # Solve the inhomogeneous equation
            dirInhom = lambda t , qDir : self.A@qDir + self.inhomPart(t,f)
            solInhom = integrate.solve_ivp( dirInhom, (self.tMin,self.tMax), self.y0, method='RK45')
            homInit = solInhom.y[:,-1]
            if(rank != size-1):
                comm.Send(homInit, dest=rank+1)
            homSum = 0
            dirHom = lambda t , qDir : self.A@qDir

            # Solve the homogeneous equations (no equations for rank=0)
            if(rank!=0):
                for block in range(1,rank+1):
                    comm.Recv(homInit,source=rank-1)
                    solHom = exponentialIntegrators.expEuler(dirHom,homInit,self.tMin,self.tMax)
                    homTime = solHom.t
                    homStates = solHom.y
                    homInterp = interpolate.interp1d(homTime,homStates)
                    homSum   += homInterp(solInhom.t)
                    if(rank!=size-1):
                        comm.Send(homStates[:,-1], dest=rank+1)
            solTot = solInhom.y + homSum
            solDir = solStruct(solInhom.t,solTot)
            qDir = interpolate.interp1d(solInhom.t,solTot,bounds_error=False,fill_value=(solDir.y[:,0],solDir.y[:,-1]))

            ## Adjoint solve ##
            # Inhomogeneous
            adjInhom = lambda t , qAdj : self.A.T@qAdj + self.costIntegranddq(self.tMin+self.tMax-t,qDir(self.tMin+self.tMax-t))
            solInHomAdj = integrate.solve_ivp( adjInhom, (self.tMin,self.tMax), self.y0, method='RK45')

            # Homogeneous
            adjHomSum = 0
            adjHomInit = solInHomAdj.y[:,-1]
            if rank != 0:
                comm.Send(adjHomInit , dest=rank-1)
            adjHom = lambda t , qAdj : self.A.T@qAdj
            if(rank!=(size-1)):
                for block in range(1,size - rank):
                    comm.Recv(adjHomInit,source=rank+1)
                    solAdjHom = exponentialIntegrators.expEuler(adjHom,adjHomInit,self.tMin,self.tMax)
                    homTime = solAdjHom.t
                    homStates = solAdjHom.y
                    adjHomInterp = interpolate.interp1d(homTime,homStates)
                    adjHomSum    += adjHomInterp(solInHomAdj.t)
                    if(rank!=0):
                        comm.Send(homStates[:,-1], dest=rank-1)
            solTotAdj   = adjHomSum + solInHomAdj.y
            solAdj = solStruct(solInHomAdj.t,solTotAdj)
            # Revert back to t from τ
            solAdj.t = self.tMin+self.tMax-solAdj.t
            solAdj.t = solAdj.t[::-1]
            solAdj.y = solAdj.y[:,::-1]

            ## Calculate the cost functional and the gradient
            cost = self.costFunctional(solDir)
            cost = comm.allreduce(cost,op=MPI.SUM)
            grad = self.calculateGradient(solAdj)
            grad = comm.allreduce(grad,op=MPI.SUM)
        return cost, grad
