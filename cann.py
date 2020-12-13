# -*- coding: utf-8 -*-
"""

@author: udits
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb

#Iapp_nn is the output from the FlyNet which is the full difference matrix between query and reference image sequences. 
#Iapp is used to extract a single row from Iapp_nn which then propagates it through CANN over time. 
#r is used as an encoded signal representing movement through the environment and rinit is its initial state.

def cann(data):
    
    Nunits = data.shape[0]+1
    tmax = Nunits+1
    Ithresh = np.ones((Nunits,))
    rinit = np.zeros((1,Nunits))
    rinit[0] = 0
    dt = 0.001
    tau = 4
    rmax = 100
    Idur = 0.4
    tvec = np.arange(0,tmax+dt,dt)
    Nt = tvec.size
    Iapp = np.zeros((Nunits,Nt))
    r = np.zeros((Nunits,Nt))
    r[:,0] = rinit

    gs = norm(loc = 0., scale = 1)
    x = np.arange(-50, 51, 1)
    y = 3.5*gs.pdf(x)

    W = 1.2*np.ones((Nunits,Nunits))

    for i in range(Nunits-1):
        for j in range(Nunits-1):
            k = (j+50-i) % (Nunits-1)
            W[i,j] = y[k]

    W[Nunits-1,Nunits-1] = 0
    W[0:Nunits-1,Nunits-1] = -1.5

    Iapp_nn = np.zeros((Nunits,Nunits))
 
    Iapp_nn[:data.shape[0],:data.shape[0]] = data
    
    for i in range(Nunits):
        Iappi = Iapp_nn[int(i),:].reshape(Nunits,1)
        noni = np.round((i+1)/dt)
        noffi = np.round((i+1+Idur)/dt)
        Iapp[:,int(noni):int(noffi)] = Iappi*np.ones((1,int(noffi-noni)))

    out = np.zeros((Nunits+1,Nunits))

    j = 0
    for i in range(1,Nt):
        I = np.matmul(W,r[:,i-1]) + Iapp[:,i-1]
        newr = r[:,i-1] + dt/tau*(I-Ithresh-r[:,i-1])
        newr = np.clip(newr, 0, rmax)
        r[:,i] = newr
        if np.mod(i,np.round(Nt/Nunits)) == 0:
            out[j,:]= r[:,i]
            j+=1
            #pdb.set_trace()


    out = out[:100, :100]
    return out

    