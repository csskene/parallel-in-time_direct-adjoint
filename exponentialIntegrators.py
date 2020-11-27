import numpy as np
import math
from scipy.linalg import expm

class solStruct:
	def __init__(self,t,y):
		self.t = t
		self.y = y

def Arnoldi(L,t,x,m):
    n = x.shape[0]
    Q = np.zeros((n,m))
    H = np.zeros((m,m))
    q = x/np.linalg.norm(x)
    Q[:,0] = q
    for i in range(1,m):
        q = L(t,Q[:,i-1])
        for j in range(i):
            H[j,i-1] = np.vdot(Q[:,j],q)
            q = q - H[j,i-1]*Q[:,j]
        H[i,i-1] = np.linalg.norm(q)
        Q[:,i] = q/H[i,i-1]
    return Q, H

def expEuler(L,q0,tMin,tMax):
    m = 20
    times  = [tMin]
    states = [q0]
    Nsteps = max(math.floor((tMax-tMin)/0.05),1)
    dt = (tMax-tMin)/Nsteps
    for i in range(Nsteps):
        Q,H = Arnoldi(L,tMin + dt*i,states[i],m)
        β = np.linalg.norm(states[i])
        states.append(β*Q@expm(H*dt)[:,0])
        times.append(tMin + dt*(i+1))
    times[-1]=tMax
    return solStruct(np.array(times),np.array(states).T)
