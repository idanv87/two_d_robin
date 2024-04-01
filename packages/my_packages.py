from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import Rbf
from tabulate import tabulate
import scipy
from shapely.geometry import Polygon as poly
import shapely.plotting

class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        return torch.linalg.norm(x-y)/(torch.linalg.norm(y)+1e-10)
    @classmethod
    def relative_L1(cls,x,y):
        return torch.nn.L1Loss()(x,y)/(torch.nn.L1Loss(y,y*0)+1e-10)
    
def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params    


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, log_path, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.path = log_path

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                self.path+'best_model.pth',
            )

def Gauss_zeidel(A, b, x, theta):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
          
            x_new[i] = (1-theta)*x[i]+ theta*(b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(A@x-b)/np.linalg.norm(b)<1e-15:
             x = x_new
             return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]
            
        x = x_new

    return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]      


class interpolation_2D:
    def __init__(self, X,Y,values):
        self.rbfi = Rbf(X, Y, values)

    def __call__(self, x,y):
        return list(map(self.rbfi,x,y  ))
    
    

def plot_table(headers, data, path=None):
    try:
       
        print(tabulate(data, headers=headers, tablefmt='orgtbl'), file=path)
    except:
        print(tabulate(data, headers=headers, tablefmt='orgtbl'))
    




def gmres(A, b, x0, nmax_iter, tol):
    b_start=b.copy()
    r = b - np.asarray(A@x0).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
        y = np.asarray(A@ q[k]).reshape(-1)

        for j in range(k + 1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b)[0]

        C=np.dot(np.asarray(q).transpose(), result) + x0
        x.append(C)
   
        if (np.linalg.norm(A@C-b_start)/np.linalg.norm(b_start))<tol:
            return C,k, np.linalg.norm(A@C-b_start)/np.linalg.norm(b_start)


    return C, k, np.linalg.norm(A@C-b_start)/np.linalg.norm(b_start)


class Plotter:
    def __init__(self,ax,headers,data_x,data_y,labels, n_figs=1, **kwargs) -> None:
        self.headers=headers
        self.data_x=data_x
        self.data_y=data_y
        if len(data_x)>1:
            self.colors=['red','blue','green','black', 'orange']
        else:
            self.colors=['black']    

        self.linestyles=['solid']*5
        self.labels=labels
        self.ax=ax
        self.kwargs=kwargs

        
    def plot_figure(self):
        self.ax.set(adjustable='box')
        for i in range(len(self.data_x)):
            self.plot_single(self.headers,[self.data_x[i],self.data_y[i]],color=self.colors[i],label=self.labels[i])
            if len(self.labels)>1 and self.labels[0] != None:
                self.ax.legend(loc='upper right')

        try:
            self.ax.set_yscale(self.kwargs['scale'])   
        except:
            pass    
        try:
            self.ax.set_title(self.kwargs['title'])
        except:
            pass    
        plt.show(block=False)        
    
    def plot_single(self,headers, data, **kwargs ):
            try:
                self.ax.plot(data[0],data[1],label=kwargs['label'],color=kwargs['color'])
            except:
                self.ax.plot(data[1],label=kwargs['label'],color=kwargs['color'])    

            self.ax.set_xlabel(headers[0])
            self.ax.set_ylabel(headers[1])
            
            
            plt.show(block=False)

    def save_figure(self,fig, path):
         fig.savefig(path, format='eps', bbox_inches='tight')
         plt.show(block=True)    




def plot_contour(ax,x,y,z):

    ngridx = 200
    ngridy = 200
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    # from scipy.interpolate import griddata
    # zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    ax.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax.contourf(xi, yi, zi, levels=20, cmap="RdBu_r")
    plt.clabel(cntr1, colors = 'k', fmt = '%2.1f', fontsize=6)
    plt.colorbar(cntr1, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    ax.set(xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)))

    plt.show()
    
def example():
    d={'a':1, 'b':2}
    return d


def Least_squares(A,b):
    x0=np.random.rand(A.shape[1])*0
    def f(x):
        return np.linalg.norm(A@x-b)+0.01*np.linalg.norm(x)
    res = scipy.optimize.minimize(f, x0, method='BFGS',
               options={'xatol': 1e-8, 'disp': False})
    
    return res.x, np.linalg.norm(A@res.x-b)/np.linalg.norm(b)

def Plot_Polygon(v,ax, **kwargs):
    polygon = poly(v)
    xe,ye = polygon.exterior.xy
    ax.plot(xe,ye, **kwargs)

class Linear_solver:
    def __init__(self, type=None, verbose=False):
        self.type=type
        self.verbose=verbose

    def __call__(self, A,b): 
        if self.type==None:
            return scipy.sparse.linalg.spsolve(A,b)
        if self.type=='gmres':
            x,j,e=gmres(A,b,b*0,nmax_iter=100, tol=1e-3)
            if self.verbose:
                print(f'gmres ended after {j} iterations with error {e}')
            return x

        # # class linear solvers
#     else:
#          x,j,e=gmres(A,b,b*0,nmax_iter=100, tol=1e-1)
#          print(f'gmres ended after {j} iterations with error {e}')
#          return x

def Restriction(x,y):
    # x is sorted  set of indices
    # y is sorted  subset of indices
    n=len(y)
    m=len(x)
    assert(len(x)>len(y))
    R=np.zeros((n,m))
    for i in range(n):
        R[i,y[i]]=1
        
    return R        
    