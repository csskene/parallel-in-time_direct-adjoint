import exponentialIntegrators
from mpi4py import MPI
import numpy as np
import sys
from scipy import integrate,interpolate

class solStruct:
	def __init__(self,t,y):
		self.t = t
		self.y = y

class paraExpIntegrator:
	def __init__(self,T,hompartDir,nonlinpartDir,hompartAdj,costIntegrand,costIntegranddq,equationdf,inhomPart,n,k):
		self.T = T
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()

		self.tMax = min(T*(1-(k/(k+1))**(rank+1))/(1-(k/(k+1))**(size)),T)
		self.tMin = T*(1-(k/(k+1))**(rank))/(1-(k/(k+1))**(size))

		self.hompartDir = hompartDir
		self.nonlinpartDir = nonlinpartDir
		self.hompartAdj = hompartAdj
		if(rank==0):
			print('## Time partition for the inhomogeneous equations ##')
		print('rank =',rank,' : ', 't∈[',self.tMin,',',self.tMax,']')
		self.inhomPart = inhomPart
		self.costIntegrand = costIntegrand
		self.costIntegranddq = costIntegranddq
		self.equationdf = equationdf
		self.y0 = np.zeros(n)

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
			## Direct
			dirInhom = lambda t , qDir : self.hompartDir(qDir) + self.nonlinpartDir(qDir) + self.inhomPart(t,f)
			solDir = integrate.solve_ivp( dirInhom, (self.tMin,self.tMax), self.y0, method='RK45')
			cost   = self.costFunctional(solDir)
			qDir   = interpolate.interp1d(solDir.t,solDir.y)

			## Adjoint
			adjInhom = lambda t , qAdj : self.hompartAdj(qAdj,qDir(self.T-t)) + self.costIntegranddq(self.T-t,qDir(self.T-t))
			solAdj = integrate.solve_ivp( adjInhom, (self.tMin,self.tMax), self.y0, method='RK45')

			# Change from τ back to t and reverse order
			solAdj.t = self.T-solAdj.t
			solAdj.t = solAdj.t[::-1]
			solAdj.y = solAdj.y[:,::-1]

			grad   = self.calculateGradient(solAdj)
		else:
			## Direct solve ##
			# Inhomogeneous (sequentially)
			dirInhom = lambda t , qDir : self.hompartDir(qDir) + self.nonlinpartDir(qDir) + self.inhomPart(t,f)
			q0 = np.zeros(self.y0.shape[0])
			if(rank==0):
				solDir = integrate.solve_ivp(dirInhom, (self.tMin,self.tMax), self.y0, method='RK45',rtol=1e-3)
				q0 = solDir.y[:,-1]
				comm.Send(q0, dest=rank+1)
			else:
				comm.Recv(q0,source=rank-1)
				solDir = integrate.solve_ivp( dirInhom, (self.tMin,self.tMax), q0, method='RK45',rtol=1e-3)
				if(rank!=size-1):
					comm.Send(solDir.y[:,-1], dest=rank+1)
			qDir = interpolate.interp1d(solDir.t,solDir.y,bounds_error=False,fill_value=(solDir.y[:,0],solDir.y[:,-1]))

			## Adjoint solve ##
			# Inhomogeneous
			adjInhom = lambda t , qAdj : self.hompartAdj(qAdj,qDir(self.tMin+self.tMax-t)) + self.costIntegranddq(self.tMin+self.tMax-t,qDir(self.tMin+self.tMax-t))
			solInHomAdj = integrate.solve_ivp( adjInhom, (self.tMin,self.tMax), self.y0, method='RK45')

			########### Redistribute for the homogeneous solve ##########
			## Redistribute the direct solution to every processor
			## Get the times and solution from every processor
			times = comm.allgather(solDir.t)
			sol   = comm.allgather(solDir.y)
			## Concatenate them together
			newSol = sol[0]
			newTimes = times[0]
			for i in range(1,size):
				newSol = np.hstack((newSol[:,:-1],sol[i]))
				newTimes = np.hstack((newTimes[:-1],times[i]))
			qDir = interpolate.interp1d(newTimes,newSol)
			###################################################

			# Homogeneous
			adjHomInit = solInHomAdj.y[:,-1]
			adjHom = lambda t , qAdj : self.hompartAdj(qAdj,qDir(self.tMin-t))
			if(rank!=0):
				solAdjHom = exponentialIntegrators.expEuler(adjHom,adjHomInit,0.,self.tMin)
			else:
				homTime = np.array([0,self.T])
				homStates = np.zeros((self.y0.shape[0],2))
				solAdjHom = solStruct(homTime ,homStates)

			###################################################
			# Gather all the homogeneous solutions and sum them
			# Revert back to t from τ
			InhomTime  = self.tMin+self.tMax-solInHomAdj.t
			if(rank==(size-1)):
				InhomTime[0] =self.T
			if(rank!=0):
				homTime = self.tMin-solAdjHom.t
			InhomStates = solInHomAdj.y
			homStates   = solAdjHom.y
			if(rank==0):
				InhomTime = np.hstack(([self.T],InhomTime[0]+1e-6,InhomTime))
				InhomStates = np.hstack((np.zeros((self.y0.shape[0],2)),InhomStates))
			elif(rank!=(size-1)):
				homTime = np.hstack(([self.T],homTime[0]+1e-6,homTime))
				homStates = np.hstack((np.zeros((self.y0.shape[0],2)),homStates))
				InhomTime = np.hstack(([self.T],InhomTime[0]+1e-6,InhomTime,InhomTime[-1]-1e-6,[0.]))
				InhomStates = np.hstack((np.zeros((self.y0.shape[0],2)),InhomStates,np.zeros((self.y0.shape[0],2))))
			elif(rank==(size-1)):
				InhomTime = np.hstack((InhomTime,InhomTime[-1]-1e-6,[0]))
				InhomStates = np.hstack((InhomStates,np.zeros((self.y0.shape[0],2))))
				homTime = np.hstack(([self.T],homTime[0]+1e-6,homTime))
				homStates = np.hstack((np.zeros((self.y0.shape[0],2)),homStates))
			# Gather and interpolate the adjoint solution
			homStatesAll = comm.allgather(homStates)
			homTimesAll = comm.allgather(homTime)
			inhomStatesAll = comm.allgather(InhomStates)
			inhomTimesAll = comm.allgather(InhomTime)
			solTotAdj = 0
			adjtVec = np.linspace(0,self.T,1000)
			for i in range(size):
				homAllInterp = interpolate.interp1d(homTimesAll[i],homStatesAll[i])
				inhomAllInterp = interpolate.interp1d(inhomTimesAll[i],inhomStatesAll[i])
				solTotAdj += inhomAllInterp(adjtVec) + homAllInterp(adjtVec)
			solAdj = solStruct(adjtVec,solTotAdj)
			###################################################

			## Calculate the cost functional and the gradient
			cost = self.costFunctional(solDir)
			cost = comm.allreduce(cost,op=MPI.SUM)
			grad = self.calculateGradient(solAdj)
			grad = comm.allreduce(grad,op=MPI.SUM)/size
		return cost, grad
