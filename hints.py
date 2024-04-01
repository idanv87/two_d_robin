import os
import sys
import math
from matplotlib.ticker import ScalarFormatter


from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf


from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D
from geometry import *
from two_d_model import geo_deeponet, Deeponet



# model=geo_deeponet( 2, 77,2, 99)



def deeponet( G1, x_domain, y_domain):
    int_points=np.vstack([x_domain,y_domain]).T
    model_r=Deeponet(2,162)
    model_c=Deeponet(2,162)
    experment_path=Constants.path+'runs/'
    best_model=torch.load(experment_path+'2024.04.01.15.36.46best_model.pth')
    model_c.load_state_dict(best_model['model_state_dict'])
    
    best_model=torch.load(experment_path+'2024.04.01.16.18.19best_model.pth')
    model_r.load_state_dict(best_model['model_state_dict'])
    

    x=np.linspace(0,0.5,10)
    y=np.linspace(0,1,20)
    domain=Rect(x,y)
    xi=(domain.X).flatten()
    yi=(domain.Y).flatten()
    
    # F=np.array(f_real(xi, yi))+1J*np.array(f_imag(xi, yi))
    F=G1
    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        a1=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.cfloat).repeat(y1.shape[0],1)
        pred2=model_r([y1, a1])+0*model_c([y1, a1])
    return pred2.numpy()
    # for j in range(len(x_domain)):
       
    #         X_test_i.append([
    #                     torch.tensor([x_domain[j],y_domain[j]], dtype=torch.float32), 
    #                      torch.tensor(F, dtype=torch.cfloat),
    #                      ])
    #         Y_test_i.append(torch.tensor(0, dtype=torch.float32))

    
    # test_dataset = SonarDataset(X_test_i, Y_test_i)
    # test_dataloader=create_loader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # coords=[]
    # prediction=[]
    # with torch.no_grad():    
    #     for input,output in test_dataloader:
    #         coords.append(input[0])
    #         prediction.append(model(input))

    # coords=np.squeeze(torch.cat(coords,axis=0).numpy())
    # prediction=torch.cat(prediction,axis=0).numpy()

    # return prediction

# def network(model, func, J, J_in, hint_init):
#     A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
#     ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")

#     b=func(x_domain, y_domain)
#     solution=scipy.sparse.linalg.spsolve(A, b)
#     predicted=deeponet(model, func)
#     # print(np.linalg.norm(solution-predicted)/np.linalg.norm(solution))


#     if hint_init:
#         x=deeponet(model, func)
#     else:
#         x=x=deeponet(model, func)*0

#     res_err=[]
#     err=[]
#     k_it=0

#     for i in range(1000):
#         x_0 = x
#         k_it += 1
#         theta=1
       
#         if (((k_it % J) in J_in) and (k_it > J_in[-1])):
            
#             factor = np.max(abs(sample[0]))/np.max(abs(A@x_0-b))
#             # factor=np.max(abs(grf(F, 1)))/np.max(abs(A@x_0-b))
#             x_temp = x_0*factor + \
#             deeponet(model, interpolation_2D(x_domain,y_domain,(b-A@x_0)*factor )) 
            
#             x=x_temp/factor
            
#             # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

#         else:    
#             x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]



       
#         print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
#         res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
#         err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
#         if (res_err[-1] < 1e-13) and (err[-1] < 1e-13):
#             return err, res_err
#         else:
#             pass


   
#     return err, res_err

# def run_hints(func, J, J_in, hint_init):
#     return network(model, func, J, J_in, hint_init)


# def plot_solution( path, eps_name):
#     e_deeponet, r_deeponet= torch.load(path)
    
#     fig3, ax3 = plt.subplots()   # should be J+1
#     fig3.suptitle(F'relative error, \mu={mu}, \sigma={sigma} ')

#     ax3.plot(e_deeponet, 'g')
#     # ax3.plot(r_deeponet,'r',label='res.err')
#     # ax3.legend()
#     ax3.set_xlabel('iteration')
#     ax3.set_ylabel('error')
#     ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}', transform=ax3.transAxes, fontsize=6,
#              ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
#     fig3.savefig(eps_name+'errors.eps', format='eps', bbox_inches='tight')
#     plt.show(block=True)
#     return 1


# torch.save(run_hints(func, J=J, J_in=[0], hint_init=True), Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt')
# # plot_solution(Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt', 'J='+str(J)+'k='+str(Constants.k)+'errors.pt')















