# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:49:13 2021

@author: Victor
"""

import numpy as np

def mandelbrot_naive(complex c, int T, int I):
    """
    Return percent iterations needed to reach thres (100 = thres not reached).
    Does not works with arrays. 
    
    """
    cdef complex z = 0
    cdef int i
    for i in range(1, I):
        z = z**2 + c
        
        if np.abs(z) > T:
            return i
    return 100


# def mandelbrot_vector(c, int threshold, int iterations):
#     cdef int rows = len(c)
#     fracs = np.zeros(c.shape)
    
    
#     for j in range(rows):
#         m = np.full((rows,1), True, dtype=bool)
#         z = np.zeros((len(c),1),dtype=complex)    
#         for i in range(1,iterations):
#             z[m]*=z[m]
            
#             z[m]+=c[m[:,0],j]
#             frac = np.greater(np.abs(z), threshold, out=np.full((rows,1), False), where=m)
#             fracs[frac[:,0],j] = (100.0 * i / iterations)     
#             m[np.abs(z) > threshold] = False
#             if (i == iterations-1):
#                 #fracs[fracs[:,j] > 1,j] -=1 ;
#                 fracs[fracs[:,j] == 0,j] = 100;    
            
#     return fracs
def mandelbrot_vector(c, int T, int I):
    z = np.zeros(c.shape, dtype = complex)
    Iota = np.full(z.shape, I)
    for i in range(1, I+1):
        z[Iota > (i-1)] = np.square(z[Iota > (i-1)]) + c[Iota > (i-1)]   # Only calculated for the ones which hasn't diverged 
        
        Iota[np.logical_and(np.abs(z) > T, Iota == I)] = i # Write iteration number to the matrix 

    return Iota