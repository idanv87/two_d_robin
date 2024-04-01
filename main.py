import numpy as np
import scipy
from scipy.linalg import circulant
from scipy.sparse import  kron, identity, csr_matrix
from scipy.stats import qmc
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import torch
from two_d_data_set import *
from two_d_model import Deeponet, geo_deeponet

# from draft import create_data, expand_function
from geometry import *
import time

from utils import count_trainable_params, extract_path_from_dir, save_uniqe, grf
from constants import Constants



# train_names=[Constants.path+'base_polygon/base_rect.pt']
# test_names=[Constants.path+'polygons/20_hole.pt']
# # test_names=[Constants.path+'base_polygon/base_rect.pt']
# # # test_names=[Constants.path+'polygons/10_115000.pt']
# # train_names=[Constants.path+'polygons/10_115000.pt']
# # train_names=list(set(extract_path_from_dir(Constants.path+'polygons/'))-set(test_names))
names=['1']
def generate_data(names,  save_path, number_samples,seed=0):
    # answer = input('Did you erase previous data? (y/n)')
    # if answer !='y':
    #     print('please erase')
    #     sys.exit()
        
    X=[]
    Y=[]
    n=20
    x_0=np.linspace(0,1,n)
    y_0=np.linspace(0,1,n)
    x=x_0[:int(0.5*(n-1))+1]
    y=y_0
    for name in enumerate(names):
        l=Constants.l
        domain=Rect(x,y)
        xi=(domain.X).flatten()
        yi=(domain.Y).flatten()
        f=grf(xi, number_samples,seed=seed )
        f1=grf(xi, number_samples,seed=1 )
        
        g1=grf(y[1:-1], number_samples,seed=1 )
        g2=grf(y[1:-1], number_samples,seed=2 )
        g=(g1-Constants.l*g2)
        
        for i in range(number_samples):
            ROB1=g[i]
            # G1=f[i]+1J*f1[i]
            A1,G1=solve_subdomain2(domain.x,domain.y,f[i],ROB1,l=Constants.l, side=1)
            s1=scipy.sparse.linalg.spsolve(A1, G1)*100
            for j in range(len(xi)):
                X1=[
                    torch.tensor([xi[j],yi[j]], dtype=torch.float32),
                    torch.tensor(G1, dtype=torch.cfloat),
                    ]
                Y1=torch.tensor((s1[j].real), dtype=torch.cfloat)
                save_uniqe([X1,Y1],save_path)
                X.append(X1)
                Y.append(Y1)
               
    return X,Y        

# 

if __name__=='__main__':
# if False:
    X,Y=generate_data(names, Constants.train_path, number_samples=300, seed=0)

    X_test, Y_test=generate_data(names,Constants.test_path,number_samples=1, seed=0)


# fig,ax=plt.subplots()
# for x in X:
#     ax.plot(x[1],'r')
# for x in X_test:
#     ax.plot(x[1],'b')


else:    
    train_data=extract_path_from_dir(Constants.train_path)
    test_data=extract_path_from_dir(Constants.test_path)
    start=time.time()
    s_train=[torch.load(f) for f in train_data]
    print(f"loading torch file take {time.time()-start}")
    s_test=[torch.load(f) for f in test_data]


    X_train=[s[0] for s in s_train]
    Y_train=[s[1] for s in s_train]
    X_test=[s[0] for s in s_test]
    Y_test=[s[1] for s in s_test]







# if __name__=='__main__':
    start=time.time()
    train_dataset = SonarDataset(X_train, Y_train)
    print(f"third loop {time.time()-start}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    start=time.time()
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    print(f"4th loop {time.time()-start}")

    train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)
    val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)

test_dataset = SonarDataset(X_test, Y_test)
test_dataloader=create_loader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

inp, out=next(iter(test_dataset))

# model=geo_deeponet( 2, inp[1].shape[0], inp[2].shape[0],inp[4].shape[0])

model=Deeponet( 2, inp[1].shape[0])

inp, out=next(iter(test_dataloader))
model(inp)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])

