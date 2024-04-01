import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
import os

from scipy import interpolate
from packages.my_packages import interpolation_2D, Restriction
from hints import deeponet
from utils import norms, calc_Robin, solve_subdomain, solve_subdomain2, rect_solver
import random
from random import gauss
import scipy
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import circulant
from scipy.sparse import block_diag
from scipy.sparse import vstack

# from jax.scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve, cg


import timeit


from constants import Constants
from scipy import signal
import matplotlib.pyplot as plt

def laplacian_matrix(x):
        dx=x[1]-x[0]
        Nx = len(x[1:-1])
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        return D2/dx/dx



n=20
x=np.linspace(0,1,n)
y=np.linspace(0,1,n)
L=csr_matrix(kron(laplacian_matrix(x), identity(len(y)-2)))+csr_matrix(kron(identity(len(x)-2), laplacian_matrix(y)))
ev,V=scipy.sparse.linalg.eigs(-L,k=10,return_eigenvectors=True,which="SR")
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 
F=(-(2*math.pi**2-Constants.k)*np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
solution=scipy.sparse.linalg.spsolve(L+Constants.k*scipy.sparse.identity(L.shape[0]),F)
g_truth=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()



x1=x[:int(0.5*(n-1))+1]
y1=y
x2=x[int(0.5*(n-1)):]
y2=y
X1, Y1 = np.meshgrid(x1[1:], y1[1:-1], indexing='ij') 
X2, Y2 = np.meshgrid(x2[:-1], y2[1:-1], indexing='ij') 


# print(np.cos(0.5)*np.sin(y1[1:-1]))

D=np.array(range(len(X.flatten())))
D1=np.array(range(len(X1.flatten())))
D2=np.array(range(len(X1.flatten())-len(y[1:-1]),len(X.flatten())))

R1=Restriction(D,D1)
R2=Restriction(D,D2)
u=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()*0
u1=R1@u
u2=R2@u

for k in range(100):
    J=2+1
    ROB1=calc_Robin(u2.real.reshape((len(x2)-1),len(y2)-2),x2[1]-x2[0],Constants.l,0)
    A1,G1=solve_subdomain2(x1,y1,R1@F,ROB1,l=Constants.l, side=1)
    res1=-A1@u1+G1
    if (k %J)==0:
    # if False:
        p1=math.sqrt(np.var(res1.real))+1e-10
        q1=math.sqrt(3.870779686559552)
        factor1=q1/p1
        p2=math.sqrt(np.var(res1.imag))+1e-10
        q2=math.sqrt(76.9394955717995)
        factor2=q2/p2
        # res1_real=interpolation_2D(X1.flatten(),Y1.flatten(),res1.real*factor1 )
        # res1_imag=interpolation_2D(X1.flatten(),Y1.flatten(),res1.imag*factor2 )
        from utils import grf
        # f=grf(R1@X.flatten(), 1,seed=0 )
        # g1=grf(y1[1:-1], 1,seed=1 )
        # g2=grf(y1[1:-1], 1,seed=2 )
        
        # ROB1=(g1[0]-Constants.l*g2[0])
        # A1,G1=solve_subdomain2(x1,y1,f[0],ROB1,l=Constants.l, side=1)
        
        s1=scipy.sparse.linalg.spsolve(A1, res1)
        # res1_real=interpolation_2D(X1.flatten(),Y1.flatten(),res1.real )
        # res1_imag=interpolation_2D(X1.flatten(),Y1.flatten(),res1.imag )
        res_tilde=res1.real*factor1+1J*res1.imag*factor2
        xx=deeponet(res_tilde, X1.flatten(), Y1.flatten())/100
        corr1=xx.real/factor1+1J*xx.imag/factor2
        # yy=scipy.sparse.linalg.spsolve(A1, res_tilde).real
        # print(np.linalg.norm(xx-yy)/np.linalg.norm(yy))
        
        
        
    else:    
        corr1=scipy.sparse.linalg.spsolve(A1, res1).real
        
    u1=u1+corr1
    err1=np.linalg.norm(u1-R1@g_truth)/np.linalg.norm(g_truth)
   
    ROB2=calc_Robin(u1.real.reshape((len(x1)-1),len(y1)-2),x1[1]-x1[0],Constants.l,1)
    A2,G2=solve_subdomain2(x2,y2,R2@F,ROB2,l=Constants.l, side=0)
    res2=-A2@u2+G2
    corr2=scipy.sparse.linalg.spsolve(A2, res2).real
    u2=u2+corr2
    err2=np.linalg.norm(u2-R2@g_truth)/np.linalg.norm(g_truth)
    print(err2)
    # print((err1+err2)/2)








if True:
    u2=(np.sin(math.pi*X2)*np.sin(math.pi*Y2))*0
    # u1=R1@u
    # u2=R2@u
    for k in range(100):
        ROB1=calc_Robin(u2.real,x2[1]-x2[0],Constants.l,0)
        # print(ROB1)
        # f=interpolation_2D(X.flatten(),Y.flatten(),F )
        # g=interpolate.InterpolatedUnivariateSpline(y2[1:-1], ROB1)
        # u=deeponet(f,g,X.flatten(),Y.flatten()).reshape((len(x1)-1),len(y1)-2)
        u1=(solve_subdomain(x1,y1,R1@F,ROB1,l=Constants.l, side=1)).reshape((len(x1)-1),len(y1)-2)
        print(u1)
        g_truth=(np.sin(math.pi*X1)*np.sin(math.pi*Y1)).flatten()
        # print(np.linalg.norm(u-u1)/np.linalg.norm(u1))
        print(np.linalg.norm(g_truth-u1.flatten())/np.linalg.norm(g_truth))
        ROB2=calc_Robin(u1.real,x1[1]-x1[0],Constants.l,1)
        u2=(solve_subdomain(x2,y2,R2@F,ROB2,l=Constants.l, side=0)).reshape((len(x2)-1),len(y2)-2)
        # print(np.linalg.norm(np.sin(math.pi*X)*np.sin(math.pi*Y)-u2)/np.linalg.norm(u2))
        

  
   
   
    
# n=2**5+1
# x=np.linspace(0,1,n)
# y=np.linspace(0,1,n)
# L=csr_matrix(kron(laplacian_matrix(x), identity(len(y)-2)))+csr_matrix(kron(identity(len(x)-2), laplacian_matrix(y)))
# ev,V=scipy.sparse.linalg.eigs(-L,k=10,return_eigenvectors=True,which="SR")
# X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 
# F=(-(2*math.pi**2-Constants.k)*np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
# solution=scipy.sparse.linalg.spsolve(L+Constants.k*scipy.sparse.identity(L.shape[0]),F)
# g_truth=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
# print(np.linalg.norm(g_truth-solution)/np.linalg.norm(g_truth))


# x1=x[:int(0.5*(n-1))+1]
# y1=y
# x2=x[int(0.5*(n-1)):]
# y2=y
# X1, Y1 = np.meshgrid(x1[1:], y1[1:-1], indexing='ij') 
# X2, Y2 = np.meshgrid(x2[:-1], y2[1:-1], indexing='ij') 

# F=grf(X.flatten(), 1, seed=0)[0]

# D=np.array(range(len(X.flatten())))
# D1=np.array(range(len(X1.flatten())))
# D2=np.array(range(len(X1.flatten())-len(y[1:-1]),len(X.flatten())))

# R1=Restriction(D,D1)
# R2=Restriction(D,D2)

# u=(np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
# # u=X.flatten()*0
# u1=R1@u
# u2=R2@u

# l=1J*np.sqrt(Constants.k)
# ROB1=calc_Robin(np.sin(math.pi*X2)*np.sin(math.pi*Y2),x2[1]-x2[0],l=l,side=0)
# X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij') 
# F=(-(2*math.pi**2-Constants.k)*np.sin(math.pi*X)*np.sin(math.pi*Y)).flatten()
# solution=scipy.sparse.linalg.spsolve(L+Constants.k*scipy.sparse.identity(L.shape[0]),F)
# x1=A@x_0-b
# p=math.sqrt(np.var(x1))
# q=math.sqrt(np.var(b))
# factor=q/p

